use crate::backend::Request;
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Deserialize)]
pub struct Document {
    pub id: String,
    pub topic: String,
    pub title: String,
    pub source: String,
    pub locator: String,
    pub context: String,
    pub questions: Vec<String>,
}

pub fn load() -> Result<Vec<Document>, String> {
    let bytes = std::fs::read(crate::root().join("demo/knowledge.json"))
        .map_err(|e| format!("资料库读取失败：{e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("资料库格式错误：{e}"))
}

fn terms(text: &str) -> BTreeSet<String> {
    let chars: Vec<char> = text.chars().filter(|c| c.is_alphanumeric()).collect();
    chars.windows(2).map(|p| p.iter().collect()).collect()
}

fn overlap(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    2.0 * a.intersection(b).count() as f32 / (a.len() + b.len()) as f32
}

pub fn matches(query: &str, topic: &str, documents: &[Document]) -> Vec<(usize, f32)> {
    let tokens = terms(query);
    if tokens.is_empty() {
        return Vec::new();
    }
    let mut hits: Vec<_> = documents
        .iter()
        .enumerate()
        .filter(|(_, d)| topic == "全部资料" || d.topic == topic)
        .map(|(index, d)| {
            let alias = d
                .questions
                .iter()
                .map(|q| overlap(&tokens, &terms(q)))
                .fold(0.0, f32::max);
            let evidence = overlap(&tokens, &terms(&d.context));
            (index, alias.max(evidence * 0.75))
        })
        .filter(|(_, score)| *score >= 0.20)
        .collect();
    hits.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    hits.truncate(3);
    hits
}

pub fn request(doc: &Document, question: String, automatic: bool) -> Request {
    Request {
        source: doc.source.clone(),
        locator: doc.locator.clone(),
        context: doc.context.clone(),
        question,
        example_id: None,
        retrieval_mode: if automatic {
            "本地关键词匹配"
        } else {
            "手动选择资料"
        }
        .into(),
        knowledge_id: Some(doc.id.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_match_has_no_reference_answer() {
        let docs = load().expect("local source snippets");
        let hits = matches(
            "合成示例课程的选课时间是什么？",
            "全部资料",
            &docs,
        );
        assert!(!hits.is_empty());
        let selected = &docs[hits[0].0];
        assert!(selected.context.contains("3月5日"));
        let json = serde_json::to_value(request(selected, "考试时间？".into(), true))
            .expect("request JSON");
        assert!(json.get("output").is_none());
        assert!(json.get("reference").is_none());
        assert!(matches("xyzqvzzakkk", "全部资料", &docs).is_empty());
    }
}
