use crate::{
    backend::{Answer, Backend, Event, Example, Job, Request},
    knowledge::{self, Document},
};
use eframe::egui::{self, Color32, FontFamily, FontId, RichText, Stroke, TextStyle};
use std::{
    collections::BTreeSet,
    fs,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

enum Action {
    Load,
    Ask,
    Example(usize),
    Topic(String),
    FreeQuestion,
    History(usize),
    Source(usize),
    OpenFolder,
    Screenshot,
    Dismiss,
}

pub struct CampusApp {
    backend: Backend,
    examples: Vec<Example>,
    documents: Vec<Document>,
    topics: Vec<String>,
    topic: String,
    matches: Vec<(usize, f32)>,
    draft: Request,
    automatic: bool,
    ready: bool,
    busy: bool,
    busy_since: Option<Instant>,
    status: String,
    notice: Option<String>,
    session: Option<PathBuf>,
    current: Option<Answer>,
    history: Vec<Answer>,
    sidebar: bool,
    load_seconds: Option<f64>,
}

fn card() -> egui::Frame {
    egui::Frame::NONE
        .fill(Color32::WHITE)
        .stroke(Stroke::new(1.0, Color32::from_gray(220)))
        .corner_radius(16)
        .inner_margin(22)
}

fn title(ui: &mut egui::Ui, text: &str) {
    ui.label(RichText::new(text).size(22.0).strong());
}
fn small(ui: &mut egui::Ui, text: impl Into<String>) {
    ui.label(RichText::new(text.into()).size(13.0));
}
fn primary(label: &str) -> egui::Button<'_> {
    egui::Button::new(RichText::new(label).strong())
        .fill(Color32::from_gray(240))
        .stroke(Stroke::new(1.5, Color32::BLACK))
        .corner_radius(9)
        .min_size(egui::vec2(132.0, 44.0))
}

impl CampusApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::from_context(cc.egui_ctx.clone())
    }
    fn from_context(ctx: egui::Context) -> Self {
        setup(&ctx);
        let mut notice = None;
        let examples = crate::backend::examples().unwrap_or_else(|e| {
            notice = Some(e);
            Vec::new()
        });
        let documents = knowledge::load().unwrap_or_else(|e| {
            notice = Some(e);
            Vec::new()
        });
        let topics = documents
            .iter()
            .map(|d| d.topic.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let draft = examples.first().map(Request::from).unwrap_or_default();
        Self {
            backend: Backend::new(ctx, false),
            examples,
            documents,
            topics,
            topic: "全部资料".into(),
            matches: Vec::new(),
            draft,
            automatic: false,
            ready: false,
            busy: false,
            busy_since: None,
            status: "模型未加载".into(),
            notice,
            session: None,
            current: None,
            history: Vec::new(),
            sidebar: true,
            load_seconds: None,
        }
    }
    fn refresh_matches(&mut self) {
        self.matches = knowledge::matches(&self.draft.question, &self.topic, &self.documents);
    }
    fn begin(&mut self, job: Job, status: &str) {
        match self.backend.send(job) {
            Ok(()) => {
                self.busy = true;
                self.busy_since = Some(Instant::now());
                self.status = status.into();
                self.notice = None;
            }
            Err(e) => self.notice = Some(e),
        }
    }
    fn ask(&mut self) {
        if self.busy || !self.ready {
            return;
        }
        if self.draft.question.trim().is_empty() {
            self.notice = Some("请先输入一个教务或教师介绍问题。".into());
            return;
        }
        if self.automatic {
            self.refresh_matches();
            if let Some(&(index, _)) = self.matches.first() {
                self.draft =
                    knowledge::request(&self.documents[index], self.draft.question.clone(), true);
            } else {
                self.notice =
                    Some("没有匹配到相关资料。请换用具体关键词，或关闭自动匹配后粘贴原文。".into());
                return;
            }
        }
        if self.draft.context.trim().is_empty() || self.draft.source.trim().is_empty() {
            self.notice = Some("请先选择资料，或填写资料出处与相关原文。".into());
            return;
        }
        self.current = None;
        self.begin(Job::Generate(self.draft.clone()), "正在生成回答");
    }
    fn actions(&mut self, actions: Vec<Action>, ctx: &egui::Context) {
        for action in actions {
            match action {
                Action::Dismiss => self.notice = None,
                Action::Load if !self.ready && !self.busy => {
                    self.begin(Job::Load, "正在加载本地模型")
                }
                Action::Ask => self.ask(),
                Action::Example(i) if !self.busy => {
                    self.draft = Request::from(&self.examples[i]);
                    self.automatic = false;
                    self.current = None;
                    self.topic = "全部资料".into();
                    self.matches.clear();
                    self.notice = None;
                }
                Action::FreeQuestion if !self.busy => {
                    self.draft = Request::default();
                    self.topic = "全部资料".into();
                    self.automatic = true;
                    self.current = None;
                    self.matches.clear();
                    self.notice = None;
                }
                Action::Topic(topic) if !self.busy => {
                    self.topic = topic;
                    self.automatic = true;
                    self.draft = Request::default();
                    self.refresh_matches();
                    self.current = None;
                }
                Action::History(i) if !self.busy => {
                    let result = &self.history[i];
                    self.draft = Request::from(result);
                    self.current = Some(result.clone());
                    self.automatic = false;
                    self.notice = None;
                }
                Action::Source(i) if !self.busy => {
                    let question = if self.draft.question.trim().is_empty() {
                        self.documents[i].title.clone()
                    } else {
                        self.draft.question.clone()
                    };
                    self.draft = knowledge::request(&self.documents[i], question, false);
                    self.automatic = false;
                    self.current = None;
                    self.notice = None;
                }
                Action::OpenFolder => {
                    if let Some(p) = &self.session
                        && let Err(e) = self.backend.send(Job::Open(p.clone()))
                    {
                        self.notice = Some(e);
                    }
                }
                Action::Screenshot => {
                    if let (Some(session), Some(record)) = (&self.session, &self.current) {
                        let stamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis();
                        let path = session
                            .join("screenshots")
                            .join(format!("record_{:02}_{stamp}.png", record.record_number));
                        ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(
                            egui::UserData::new(path),
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    fn receive(&mut self) {
        while let Ok(event) = self.backend.rx.try_recv() {
            match event {
                Event::Started { session_dir } => self.session = Some(session_dir.into()),
                Event::Ready { model_load_seconds } => {
                    self.ready = true;
                    self.busy = false;
                    self.busy_since = None;
                    self.load_seconds = Some(model_load_seconds);
                    self.status = "模型已就绪".into();
                }
                Event::Answer { record } => {
                    let record = *record;
                    self.status = format!("第{}条回答已保存", record.record_number);
                    self.busy = false;
                    self.busy_since = None;
                    self.current = Some(record.clone());
                    self.history.push(record);
                }
                Event::Error { message } => {
                    self.busy = false;
                    self.busy_since = None;
                    self.status = "请检查输入后重试".into();
                    self.notice = Some(message);
                }
                Event::Fatal { message } => {
                    self.ready = false;
                    self.busy = false;
                    self.busy_since = None;
                    self.status = "模型未就绪".into();
                    self.notice = Some(message);
                }
                Event::Notice { message } => self.notice = Some(message),
                Event::Closed => {
                    self.ready = false;
                    self.busy = false;
                }
            }
        }
    }
    fn sidebar_ui(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        egui::ScrollArea::vertical()
            .id_salt("navigation_scroll")
            .show(ui, |ui| {
                ui.add_space(4.0);
                title(ui, "校园服务");
                ui.add_space(16.0);
                if ui
                    .add_enabled(
                        !self.busy,
                        egui::Button::new("自由提问")
                            .min_size(egui::vec2(ui.available_width(), 38.0)),
                    )
                    .clicked()
                {
                    actions.push(Action::FreeQuestion);
                }
                ui.add_space(22.0);
                small(ui, "常见问题 / 演示案例");
                ui.add_space(8.0);
                for (index, example) in self.examples.iter().enumerate() {
                    ui.push_id(&example.id, |ui| {
                        let active = self.draft.example_id.as_deref() == Some(&example.id);
                        let text = match index {
                            0 => "教务时间查询",
                            1 => "教师研究方向",
                            2 => "资料不足问题",
                            _ => "计算能力局限",
                        };
                        if ui
                            .add_enabled(
                                !self.busy,
                                egui::Button::new(text)
                                    .selected(active)
                                    .min_size(egui::vec2(ui.available_width(), 42.0)),
                            )
                            .on_hover_text(&example.title)
                            .clicked()
                        {
                            actions.push(Action::Example(index));
                        }
                    });
                    ui.add_space(4.0);
                }
                ui.add_space(20.0);
                small(ui, "资料分类");
                ui.add_space(8.0);
                for topic in std::iter::once(&"全部资料".to_owned()).chain(self.topics.iter()) {
                    if ui
                        .add_enabled(
                            !self.busy,
                            egui::Button::new(topic)
                                .selected(self.topic == *topic)
                                .frame_when_inactive(false)
                                .min_size(egui::vec2(ui.available_width(), 30.0)),
                        )
                        .clicked()
                    {
                        actions.push(Action::Topic(topic.clone()));
                    }
                }
                ui.add_space(20.0);
                egui::CollapsingHeader::new(format!("本次历史 · {}条", self.history.len()))
                    .id_salt("history")
                    .show(ui, |ui| {
                        if self.history.is_empty() {
                            small(ui, "完成问答后自动记录。 ");
                        }
                        for (index, record) in self.history.iter().enumerate().rev() {
                            ui.push_id(record.record_number, |ui| {
                                let label = format!(
                                    "{}. {}",
                                    record.record_number,
                                    record.question.chars().take(20).collect::<String>()
                                );
                                if ui
                                    .add_enabled(!self.busy, egui::Button::new(label).frame(false))
                                    .clicked()
                                {
                                    actions.push(Action::History(index));
                                }
                            });
                        }
                    });
                ui.add_space(16.0);
                egui::CollapsingHeader::new("模型与资料信息")
                    .id_salt("model_info")
                    .show(ui, |ui| {
                        small(
                            ui,
                            "Qwen2.5-0.5B-Instruct\nQ4适配器 · rank 4\n本地4bit QLoRA推理",
                        );
                        small(
                            ui,
                            format!(
                                "{}个来源片段\n{}个资料主题",
                                self.documents.len(),
                                self.topics.len()
                            ),
                        );
                        if let Some(seconds) = self.load_seconds {
                            small(ui, format!("模型加载 {seconds:.1}秒"));
                        }
                        small(ui, "历史资料版本；并非学校官方在线服务。");
                    });
            });
    }
    fn question_ui(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            title(ui, "今天想了解什么？");
            ui.add_space(8.0);
            small(ui, "教务规则、学生办事与教师介绍，先依据资料再回答。");
            if ui.ctx().content_rect().width() < 1000.0 {
                egui::ComboBox::from_id_salt("compact_examples")
                    .selected_text("选择常见问题")
                    .show_ui(ui, |ui| {
                        for (index, example) in self.examples.iter().enumerate() {
                            if ui.selectable_label(false, &example.title).clicked() {
                                actions.push(Action::Example(index));
                            }
                        }
                        if ui.selectable_label(false, "自由提问").clicked() {
                            actions.push(Action::FreeQuestion);
                        }
                    });
            }
            ui.add_space(16.0);
            let response = ui.add_enabled(
                !self.busy,
                egui::TextEdit::multiline(&mut self.draft.question)
                    .id_salt("question_editor")
                    .hint_text("例如：已有未通过成绩的课，应该选择补修还是重修？")
                    .desired_width(f32::INFINITY)
                    .desired_rows(3)
                    .font(TextStyle::Body),
            );
            let mut changed = response.changed();
            ui.add_space(12.0);
            let checkbox = ui.add_enabled(
                !self.busy,
                egui::Checkbox::new(&mut self.automatic, "自动匹配本地资料"),
            );
            if checkbox.changed() {
                changed = true;
            }
            if changed {
                self.current = None;
                self.draft.example_id = None;
                if self.automatic {
                    self.refresh_matches();
                    if let Some(&(i, _)) = self.matches.first() {
                        self.draft = knowledge::request(
                            &self.documents[i],
                            self.draft.question.clone(),
                            true,
                        );
                    } else {
                        self.draft = Request {
                            question: self.draft.question.clone(),
                            retrieval_mode: "本地关键词匹配".into(),
                            ..Default::default()
                        };
                    }
                } else {
                    self.draft.retrieval_mode = "手动选择资料".into();
                }
            }
            ui.add_space(14.0);
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(
                        !self.ready && !self.busy,
                        if self.ready {
                            egui::Button::new("Q4 已就绪").min_size(egui::vec2(132.0, 44.0))
                        } else {
                            primary("加载模型")
                        },
                    )
                    .on_disabled_hover_text("模型已加载或当前正在运行。")
                    .clicked()
                {
                    actions.push(Action::Load);
                }
                if ui
                    .add_enabled(
                        self.ready && !self.busy,
                        if self.ready {
                            primary("生成回答")
                        } else {
                            egui::Button::new("加载后可提问").min_size(egui::vec2(132.0, 44.0))
                        },
                    )
                    .on_disabled_hover_text("请先加载模型，等待当前操作完成。")
                    .clicked()
                {
                    actions.push(Action::Ask);
                }
                if self.busy {
                    ui.spinner();
                    if let Some(t) = self.busy_since {
                        small(ui, format!("已等待 {:.0}秒", t.elapsed().as_secs_f32()));
                    }
                }
            });
            ui.add_space(6.0);
            small(
                ui,
                if self.automatic {
                    "匹配结果可能有误，请核对下面的原文。无匹配时请换关键词或手动提供资料。"
                } else {
                    "当前使用下方选定资料。可改写问题；更换主题时请同时更换相应资料。"
                },
            );
        });
        ui.add_space(16.0);
        card().show(ui, |ui| {
            title(ui, "参考资料");
            ui.add_space(10.0);
            let mut changed = false;
            small(ui, "资料出处");
            changed |= ui
                .add_enabled(
                    !self.busy && !self.automatic,
                    egui::TextEdit::singleline(&mut self.draft.source)
                        .id_salt("source_name")
                        .desired_width(f32::INFINITY)
                        .hint_text("选择原文件，或标注自拟演示资料"),
                )
                .changed();
            ui.add_space(8.0);
            egui::ScrollArea::vertical()
                .id_salt("context_scroll")
                .max_height(210.0)
                .show(ui, |ui| {
                    changed |= ui
                        .add_enabled(
                            !self.busy && !self.automatic,
                            egui::TextEdit::multiline(&mut self.draft.context)
                                .id_salt("source_context")
                                .desired_width(f32::INFINITY)
                                .desired_rows(7)
                                .font(TextStyle::Body)
                                .hint_text("只保留与问题相关的原文，不粘贴参考答案。"),
                        )
                        .changed();
                });
            ui.add_space(10.0);
            egui::CollapsingHeader::new("原文定位与其他候选")
                .id_salt("source_details")
                .show(ui, |ui| {
                    changed |= ui
                        .add_enabled(
                            !self.busy && !self.automatic,
                            egui::TextEdit::singleline(&mut self.draft.locator)
                                .id_salt("source_location")
                                .desired_width(f32::INFINITY)
                                .hint_text("页码、段落或表格位置"),
                        )
                        .changed();
                    for &(index, _) in &self.matches {
                        let doc = &self.documents[index];
                        ui.push_id(&doc.id, |ui| {
                            if ui
                                .add_enabled(
                                    !self.busy,
                                    egui::Button::new(format!("{} · {}", doc.topic, doc.title)),
                                )
                                .clicked()
                            {
                                actions.push(Action::Source(index));
                            }
                        });
                    }
                    if self.matches.is_empty() {
                        for (index, doc) in self
                            .documents
                            .iter()
                            .enumerate()
                            .filter(|(_, d)| self.topic == "全部资料" || d.topic == self.topic)
                            .take(5)
                        {
                            ui.push_id(&doc.id, |ui| {
                                if ui
                                    .add_enabled(!self.busy, egui::Button::new(&doc.title))
                                    .clicked()
                                {
                                    actions.push(Action::Source(index));
                                }
                            });
                        }
                    }
                });
            if changed {
                self.current = None;
                self.draft.example_id = None;
                self.draft.knowledge_id = None;
                self.draft.retrieval_mode = "手动提供资料".into();
            }
            ui.add_space(8.0);
            small(
                ui,
                "金额、日期和学分以原材料版本为准，现行安排请查看学校最新通知。",
            );
        });
    }
    fn answer_ui(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal_wrapped(|ui| {
                title(ui, "问答结果");
                if let Some(record) = &self.current {
                    ui.label(
                        RichText::new(format!("已保存 · 第{}条", record.record_number)).size(13.0),
                    );
                }
            });
            ui.add_space(18.0);
            if let Some(record) = &self.current {
                egui::ScrollArea::vertical()
                    .id_salt("answer_scroll")
                    .min_scrolled_height(240.0)
                    .max_height(380.0)
                    .show(ui, |ui| {
                        ui.label(RichText::new(&record.prediction).size(19.0));
                    });
                ui.add_space(20.0);
                ui.separator();
                ui.add_space(16.0);
                ui.horizontal_wrapped(|ui| {
                    ui.vertical(|ui| {
                        small(ui, "生成用时");
                        ui.label(
                            RichText::new(format!("{:.2} 秒", record.generate_seconds))
                                .size(22.0)
                                .strong(),
                        );
                    });
                    ui.add_space(24.0);
                    ui.vertical(|ui| {
                        small(ui, "生成长度");
                        ui.label(
                            RichText::new(format!(
                                "{} token",
                                record.generated_tokens_including_stop
                            ))
                            .size(22.0)
                            .strong(),
                        );
                    });
                });
                ui.add_space(12.0);
                small(
                    ui,
                    if record.reached_token_limit {
                        "回答达到256生成上限，可能未完成；请在核验记录中注明。"
                    } else {
                        "模型原始回答已保留，尚需核对事实与适用条件。"
                    },
                );
                if record.cold_first_answer {
                    small(ui, "首次回答未预热，耗时仅作演示记录。");
                }
                ui.add_space(18.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("复制回答").clicked() {
                        ui.ctx().copy_text(record.prediction.clone());
                    }
                    if ui
                        .add_enabled(!self.busy, egui::Button::new("保存界面截图"))
                        .clicked()
                    {
                        actions.push(Action::Screenshot);
                    }
                });
                ui.add_space(14.0);
                small(ui, "回答依据");
                small(ui, &record.source);
                small(ui, &record.locator);
            } else {
                ui.add_space(35.0);
                ui.label(
                    RichText::new(if self.busy {
                        "正在准备回答…"
                    } else {
                        "回答将显示在这里"
                    })
                    .size(22.0),
                );
                ui.add_space(12.0);
                ui.label(if self.busy {
                    "你可以在这里等待，完成后会自动保存原始回答与记录。"
                } else {
                    "加载模型，选择常见问题或输入自己的问题，再点击生成回答。"
                });
                ui.add_space(48.0);
                ui.separator();
                ui.add_space(16.0);
                small(
                    ui,
                    "资料不足时应明确无法确认。模型可能出错，建议保留一个错误案例用于论文讨论。",
                );
                ui.add_space(22.0);
            }
        });
        ui.add_space(16.0);
        egui::Frame::NONE.inner_margin(8).show(ui, |ui| {
            small(ui, "演示记录自动保存");
            small(
                ui,
                "你只需核对答案、保存案例截图，并填写记录目录中的人工记录.txt。",
            );
            if ui
                .add_enabled(self.session.is_some(), egui::Button::new("打开记录文件夹"))
                .clicked()
            {
                actions.push(Action::OpenFolder);
            }
        });
    }
}

impl eframe::App for CampusApp {
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = self.backend.shutdown();
    }
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.receive();
        let ctx = ui.ctx().clone();
        let mut actions = Vec::new();
        let screenshots = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|event| {
                    if let egui::Event::Screenshot {
                        image, user_data, ..
                    } = event
                    {
                        user_data
                            .data
                            .as_ref()
                            .and_then(|d| d.downcast_ref::<PathBuf>())
                            .map(|p| (image.clone(), p.clone()))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        });
        for (image, path) in screenshots {
            self.backend.save_screenshot(image, path);
        }
        if ctx.input(|i| i.viewport().close_requested()) && self.busy {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.notice = Some("当前正在加载或生成。请等待保存完成后关闭窗口。".into());
        }
        if ui.input_mut(|i| i.consume_key(egui::Modifiers::CTRL, egui::Key::Enter)) {
            actions.push(Action::Ask);
        }
        egui::Panel::top("header")
            .frame(
                egui::Frame::NONE
                    .fill(Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(28, 20)),
            )
            .show(ui, |ui| {
                let width = ui.available_width();
                if width < 1000.0 {
                    ui.vertical(|ui| {
                        ui.label(RichText::new("校园资料问答系统").size(30.0).strong());
                        ui.add_space(4.0);
                        small(ui, "基于校园资料的本地问答 · 教务服务与教师介绍");
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new(format!("Q4 · {}", self.status)).strong());
                            if width >= 790.0
                                && ui
                                    .button(if self.sidebar {
                                        "收起导航"
                                    } else {
                                        "展开导航"
                                    })
                                    .clicked()
                            {
                                self.sidebar = !self.sidebar;
                            }
                        });
                    });
                } else {
                    ui.horizontal_top(|ui| {
                        ui.vertical(|ui| {
                            ui.set_width(width - 270.0);
                            ui.label(RichText::new("校园资料问答系统").size(30.0).strong());
                            ui.add_space(4.0);
                            small(ui, "基于校园资料的本地问答 · 教务服务与教师介绍");
                        });
                        ui.vertical(|ui| {
                            ui.label(RichText::new(format!("Q4 · {}", self.status)).strong());
                            if ui
                                .button(if self.sidebar {
                                    "收起导航"
                                } else {
                                    "展开导航"
                                })
                                .clicked()
                            {
                                self.sidebar = !self.sidebar;
                            }
                        });
                    });
                }
            });
        egui::Panel::bottom("footer")
            .frame(
                egui::Frame::NONE
                    .fill(Color32::WHITE)
                    .inner_margin(egui::Margin::symmetric(24, 10)),
            )
            .show(ui, |ui| {
                small(
                    ui,
                    "历史资料供演示核验；本系统为毕业论文原型，非学校官方服务。",
                );
                if let Some(path) = &self.session {
                    small(ui, format!("记录目录：{}", path.display()));
                }
            });
        let narrow = ui.available_width() < 850.0;
        let mut sidebar = self.sidebar && !narrow;
        egui::Panel::left("navigation")
            .default_size(220.0)
            .min_size(185.0)
            .max_size(260.0)
            .resizable(true)
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_gray(247))
                    .inner_margin(18),
            )
            .show_collapsible(ui, &mut sidebar, |ui| self.sidebar_ui(ui, &mut actions));
        if !narrow {
            self.sidebar = sidebar;
        }
        egui::CentralPanel::default()
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_gray(247))
                    .inner_margin(24),
            )
            .show(ui, |ui| {
                if let Some(message) = &self.notice {
                    card().show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(message);
                            if ui.button("知道了").clicked() {
                                actions.push(Action::Dismiss);
                            }
                        });
                    });
                    ui.add_space(12.0);
                }
                egui::ScrollArea::vertical()
                    .id_salt("workspace_scroll")
                    .show(ui, |ui| {
                        if ui.available_width() >= 930.0 {
                            let width = ui.available_width();
                            let left = (width - 20.0) * 0.57;
                            ui.horizontal_top(|ui| {
                                ui.allocate_ui_with_layout(
                                    egui::vec2(left, 0.0),
                                    egui::Layout::top_down(egui::Align::Min),
                                    |ui| self.question_ui(ui, &mut actions),
                                );
                                ui.add_space(8.0);
                                ui.allocate_ui_with_layout(
                                    egui::vec2((width - left - 20.0).max(320.0), 0.0),
                                    egui::Layout::top_down(egui::Align::Min),
                                    |ui| self.answer_ui(ui, &mut actions),
                                );
                            });
                        } else {
                            self.question_ui(ui, &mut actions);
                            ui.add_space(16.0);
                            self.answer_ui(ui, &mut actions);
                        }
                    });
            });
        self.actions(actions, &ctx);
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

fn setup(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    if let Ok(bytes) = fs::read(r"C:\Windows\Fonts\msyh.ttc") {
        fonts
            .font_data
            .insert("chinese".into(), egui::FontData::from_owned(bytes).into());
        fonts
            .families
            .entry(FontFamily::Proportional)
            .or_default()
            .insert(0, "chinese".into());
        fonts
            .families
            .entry(FontFamily::Monospace)
            .or_default()
            .push("chinese".into());
    }
    ctx.set_fonts(fonts);
    ctx.set_theme(egui::Theme::Light);
    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, FontId::proportional(22.0)),
            (TextStyle::Body, FontId::proportional(16.0)),
            (TextStyle::Button, FontId::proportional(16.0)),
            (TextStyle::Small, FontId::proportional(13.0)),
            (TextStyle::Monospace, FontId::monospace(14.0)),
        ]
        .into();
        style.spacing.item_spacing = egui::vec2(10.0, 8.0);
        style.spacing.button_padding = egui::vec2(14.0, 9.0);
        style.animation_time = 0.12;
        let v = &mut style.visuals;
        *v = egui::Visuals::light();
        v.override_text_color = Some(Color32::BLACK);
        v.weak_text_color = Some(Color32::BLACK);
        v.hyperlink_color = Color32::BLACK;
        v.warn_fg_color = Color32::BLACK;
        v.error_fg_color = Color32::BLACK;
        v.disabled_alpha = 1.0;
        v.panel_fill = Color32::WHITE;
        v.window_fill = Color32::WHITE;
        v.extreme_bg_color = Color32::from_gray(250);
        v.text_edit_bg_color = Some(Color32::from_gray(250));
        v.faint_bg_color = Color32::from_gray(247);
        v.selection.bg_fill = Color32::from_gray(228);
        v.selection.stroke = Stroke::new(1.5, Color32::BLACK);
        for visual in [
            &mut v.widgets.noninteractive,
            &mut v.widgets.inactive,
            &mut v.widgets.hovered,
            &mut v.widgets.active,
            &mut v.widgets.open,
        ] {
            visual.fg_stroke = Stroke::new(1.0, Color32::BLACK);
            visual.corner_radius = egui::CornerRadius::same(8);
            visual.bg_fill = Color32::from_gray(247);
            visual.weak_bg_fill = Color32::from_gray(247);
            visual.bg_stroke = Stroke::new(1.0, Color32::from_gray(215));
        }
        v.widgets.hovered.bg_stroke = Stroke::new(1.5, Color32::BLACK);
        v.widgets.active.bg_stroke = Stroke::new(2.0, Color32::BLACK);
    });
}
