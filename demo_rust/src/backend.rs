use eframe::egui;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    os::windows::process::CommandExt,
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc,
        mpsc::{self, Receiver, SyncSender},
    },
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

#[derive(Clone, Deserialize)]
pub struct Example {
    pub id: String,
    pub title: String,
    pub source: String,
    pub locator: String,
    pub context: String,
    pub question: String,
}

#[derive(Clone, Default, Serialize)]
pub struct Request {
    pub source: String,
    pub locator: String,
    pub context: String,
    pub question: String,
    pub example_id: Option<String>,
    pub retrieval_mode: String,
    pub knowledge_id: Option<String>,
}

impl From<&Example> for Request {
    fn from(item: &Example) -> Self {
        Self {
            source: item.source.clone(),
            locator: item.locator.clone(),
            context: item.context.clone(),
            question: item.question.clone(),
            example_id: Some(item.id.clone()),
            retrieval_mode: "固定演示样例".into(),
            knowledge_id: None,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Answer {
    pub source: String,
    pub locator: String,
    pub context: String,
    pub question: String,
    pub example_id: Option<String>,
    pub prediction: String,
    pub record_number: usize,
    pub generate_seconds: f64,
    pub prompt_tokens: usize,
    pub generated_tokens_including_stop: usize,
    pub reached_token_limit: bool,
    pub cold_first_answer: bool,
    #[serde(default)]
    pub retrieval_mode: String,
    #[serde(default)]
    pub knowledge_id: Option<String>,
}

impl From<&Answer> for Request {
    fn from(item: &Answer) -> Self {
        Self {
            source: item.source.clone(),
            locator: item.locator.clone(),
            context: item.context.clone(),
            question: item.question.clone(),
            example_id: item.example_id.clone(),
            retrieval_mode: item.retrieval_mode.clone(),
            knowledge_id: item.knowledge_id.clone(),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Started { session_dir: String },
    Ready { model_load_seconds: f64 },
    Answer { record: Box<Answer> },
    Error { message: String },
    Fatal { message: String },
    Closed,
    Notice { message: String },
}

pub enum Job {
    Load,
    Generate(Request),
    Open(PathBuf),
    Shutdown,
}

pub struct Backend {
    tx: SyncSender<Job>,
    pub rx: Receiver<Event>,
    events: mpsc::Sender<Event>,
    ctx: egui::Context,
}

fn notify(tx: &mpsc::Sender<Event>, ctx: &egui::Context, event: Event) {
    let _ = tx.send(event);
    ctx.request_repaint();
}

impl Backend {
    pub fn new(ctx: egui::Context, verification: bool) -> Self {
        let (tx, jobs) = mpsc::sync_channel(4);
        let (events, rx) = mpsc::channel();
        let worker_events = events.clone();
        let worker_ctx = ctx.clone();
        thread::spawn(move || {
            let mut pipe: Option<Pipe> = None;
            while let Ok(job) = jobs.recv() {
                match job {
                    Job::Load => match Pipe::start(verification) {
                        Ok(mut new_pipe) => loop {
                            match new_pipe.receive() {
                                Ok(Event::Started { session_dir }) => notify(
                                    &worker_events,
                                    &worker_ctx,
                                    Event::Started { session_dir },
                                ),
                                Ok(Event::Ready { model_load_seconds }) => {
                                    pipe = Some(new_pipe);
                                    notify(
                                        &worker_events,
                                        &worker_ctx,
                                        Event::Ready { model_load_seconds },
                                    );
                                    break;
                                }
                                Ok(Event::Fatal { message }) | Ok(Event::Error { message }) => {
                                    notify(&worker_events, &worker_ctx, Event::Fatal { message });
                                    break;
                                }
                                Ok(_) => {}
                                Err(message) => {
                                    notify(&worker_events, &worker_ctx, Event::Fatal { message });
                                    break;
                                }
                            }
                        },
                        Err(message) => {
                            notify(&worker_events, &worker_ctx, Event::Fatal { message })
                        }
                    },
                    Job::Generate(request) => {
                        let response = pipe
                            .as_mut()
                            .ok_or_else(|| "请先加载模型。".to_owned())
                            .and_then(|p| p.generate(&request));
                        match response {
                            Ok(event) => notify(&worker_events, &worker_ctx, event),
                            Err(message) => {
                                pipe = None;
                                notify(&worker_events, &worker_ctx, Event::Fatal { message });
                            }
                        }
                    }
                    Job::Open(path) => {
                        let result = Command::new("explorer.exe")
                            .arg(path)
                            .creation_flags(CREATE_NO_WINDOW)
                            .spawn();
                        if let Err(e) = result {
                            notify(
                                &worker_events,
                                &worker_ctx,
                                Event::Notice {
                                    message: format!(
                                        "无法打开文件夹：{e}。请复制底部路径手动打开。"
                                    ),
                                },
                            );
                        }
                    }
                    Job::Shutdown => break,
                }
            }
            if let Some(mut p) = pipe {
                let _ = p.send(&serde_json::json!({"action":"shutdown"}));
                let _ = p.receive();
                let _ = p.child.wait();
            }
            notify(&worker_events, &worker_ctx, Event::Closed);
        });
        Self {
            tx,
            rx,
            events,
            ctx,
        }
    }

    pub fn send(&self, job: Job) -> Result<(), String> {
        self.tx
            .try_send(job)
            .map_err(|e| format!("操作未能开始：{e}。请等待当前操作结束后重试。"))
    }

    pub fn shutdown(&self) -> Result<(), String> {
        self.send(Job::Shutdown)?;
        loop {
            match self.rx.recv_timeout(Duration::from_secs(15)) {
                Ok(Event::Closed) => return Ok(()),
                Ok(_) => {}
                Err(error) => return Err(format!("推理进程退出确认失败：{error}")),
            }
        }
    }

    pub fn save_screenshot(&self, image: Arc<egui::ColorImage>, path: PathBuf) {
        let tx = self.events.clone();
        let ctx = self.ctx.clone();
        thread::spawn(move || {
            let bytes: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
            let temporary = path.with_extension("png.part");
            let result = image::save_buffer_with_format(
                &temporary,
                &bytes,
                image.width() as u32,
                image.height() as u32,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            );
            let message = match result {
                Ok(()) => match fs::rename(&temporary, &path) {
                    Ok(()) => format!("界面截图已保存：{}", path.display()),
                    Err(error) => format!("截图保存失败：{error}。可使用Win+Shift+S另行截图。"),
                },
                Err(error) => format!("截图保存失败：{error}。可使用Win+Shift+S另行截图。"),
            };
            notify(&tx, &ctx, Event::Notice { message });
        });
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        let _ = self.tx.try_send(Job::Shutdown);
    }
}

struct Pipe {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    error_file: PathBuf,
}

impl Pipe {
    fn start(verification: bool) -> Result<Self, String> {
        let directory = crate::root().join("reports/rust_demo_backend");
        fs::create_dir_all(&directory).map_err(|e| format!("无法创建运行日志：{e}"))?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos();
        let error_file = directory.join(format!("backend_{stamp}.log"));
        let log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&error_file)
            .map_err(|e| e.to_string())?;
        let executable = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut child = Command::new(std::env::var_os("QLORA_PYTHON").unwrap_or_else(|| "python".into()))
            .args(["-X", "utf8", "-u"])
            .arg(crate::root().join("scripts/demo_q4_bridge.py"))
            .arg("--purpose")
            .arg(if verification {
                "assistant_smoke_check"
            } else {
                "manual_demo"
            })
            .arg("--frontend-exe")
            .arg(executable)
            .current_dir(crate::root())
            .env("PYTHONIOENCODING", "utf-8")
            .creation_flags(CREATE_NO_WINDOW)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .spawn()
            .map_err(|e| {
                format!("无法启动Q4推理程序：{e}。请检查QLORA_PYTHON是否指向所需环境。")
            })?;
        let input = child.stdin.take().ok_or("无法打开推理输入管道")?;
        let output = child.stdout.take().ok_or("无法打开推理输出管道")?;
        Ok(Self {
            child,
            input,
            output: BufReader::new(output),
            error_file,
        })
    }
    fn send(&mut self, value: &serde_json::Value) -> Result<(), String> {
        serde_json::to_writer(&mut self.input, value).map_err(|e| e.to_string())?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("发送问题失败：{e}"))
    }
    fn receive(&mut self) -> Result<Event, String> {
        let mut line = String::new();
        self.output
            .read_line(&mut line)
            .map_err(|e| format!("读取回答失败：{e}"))?;
        if line.trim().is_empty() {
            return Err(format!(
                "推理进程已停止。请查看{}，再重新加载模型。",
                self.error_file.display()
            ));
        }
        serde_json::from_str(&line)
            .map_err(|e| format!("回答格式异常：{e}。运行日志：{}", self.error_file.display()))
    }
    fn generate(&mut self, request: &Request) -> Result<Event, String> {
        let mut message = serde_json::to_value(request).map_err(|e| e.to_string())?;
        message["action"] = serde_json::json!("generate");
        self.send(&message)?;
        self.receive()
    }
}

impl Drop for Pipe {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn examples() -> Result<Vec<Example>, String> {
    let bytes = fs::read(crate::root().join("demo/examples.json"))
        .map_err(|e| format!("无法读取演示样例：{e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("演示样例格式错误：{e}"))
}

pub fn verify_backend() -> Result<(), String> {
    let backend = Backend::new(egui::Context::default(), true);
    backend.send(Job::Load)?;
    let mut session = None;
    loop {
        match backend
            .rx
            .recv_timeout(Duration::from_secs(120))
            .map_err(|e| e.to_string())?
        {
            Event::Started { session_dir } => session = Some(session_dir),
            Event::Ready { .. } => break,
            Event::Fatal { message } => return Err(message),
            _ => {}
        }
    }
    let mut records = Vec::new();
    for example in examples()? {
        backend.send(Job::Generate(Request::from(&example)))?;
        match backend
            .rx
            .recv_timeout(Duration::from_secs(120))
            .map_err(|e| e.to_string())?
        {
            Event::Answer { record } if !record.prediction.trim().is_empty() => {
                records.push(record)
            }
            Event::Error { message } | Event::Fatal { message } => return Err(message),
            _ => return Err("预检收到非回答事件".into()),
        }
    }
    let documents = crate::knowledge::load()?;
    let question = "合成示例课程的选课时间是什么？";
    let hits = crate::knowledge::matches(question, "全部资料", &documents);
    let index = hits.first().ok_or("自动资料匹配预检未找到片段")?.0;
    let request = crate::knowledge::request(&documents[index], question.into(), true);
    backend.send(Job::Generate(request))?;
    match backend
        .rx
        .recv_timeout(Duration::from_secs(120))
        .map_err(|e| e.to_string())?
    {
        Event::Answer { record }
            if record.knowledge_id.as_deref() == Some(&documents[index].id)
                && !record.prediction.trim().is_empty() =>
        {
            records.push(record)
        }
        Event::Error { message } | Event::Fatal { message } => return Err(message),
        _ => return Err("自动匹配生成预检失败".into()),
    }
    let long = Request {
        source: "自拟预检资料".into(),
        context: "超长资料".repeat(3000),
        question: "测试".into(),
        ..Default::default()
    };
    backend.send(Job::Generate(long))?;
    match backend
        .rx
        .recv_timeout(Duration::from_secs(30))
        .map_err(|e| e.to_string())?
    {
        Event::Error { message } if message.contains("512") => {}
        _ => return Err("未拒绝过长输入".into()),
    }
    let report = serde_json::json!({"status":"passed","interface":"Rust eframe 0.36.2","session_dir":session,"records":records,"long_input_rejected":true,"purpose":"assistant_smoke_check"});
    fs::write(
        crate::root().join("reports/rust_demo_verification_20261014.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    backend.shutdown()?;
    Ok(())
}
