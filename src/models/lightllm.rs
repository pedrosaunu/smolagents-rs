use std::collections::HashMap;

use anyhow::Result;
use reqwest::blocking::Client;
use serde_json::json;

use crate::{
    errors::AgentError,
    models::{
        model_traits::{Model, ModelResponse},
        openai::{AssistantMessage, Choice, OpenAIResponse},
        types::{Message, MessageRole},
    },
    tools::ToolInfo,
};

#[derive(Debug, Clone)]
pub struct LightLLMModel {
    pub base_url: String,
    pub model_id: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: Option<String>,
}

impl LightLLMModel {
    pub fn new(
        base_url: Option<&str>,
        model_id: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
    ) -> Self {
        let base_url = base_url.unwrap_or("http://localhost:8080/v1/chat/completions");
        LightLLMModel {
            base_url: base_url.to_string(),
            model_id: model_id.unwrap_or("gpt-3.5-turbo").to_string(),
            client: Client::new(),
            temperature: temperature.unwrap_or(0.5),
            api_key: api_key.or_else(|| std::env::var("LIGHTLLM_API_KEY").ok()),
        }
    }
}

impl Model for LightLLMModel {
    fn run(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);
        let messages = messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content
                })
            })
            .collect::<Vec<_>>();
        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("required");
        }
        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }
        let mut request = self.client.post(&self.base_url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }
        let response = request.send().map_err(|e| {
            AgentError::Generation(format!("Failed to get response from LightLLM: {}", e))
        })?;
        if response.status().is_success() {
            let resp: OpenAIResponse = response
                .json()
                .map_err(|e| AgentError::Generation(e.to_string()))?;
            Ok(Box::new(resp))
        } else {
            Err(AgentError::Generation(format!(
                "Failed to get response from LightLLM: {}",
                response.text().unwrap_or_default()
            )))
        }
    }

    fn run_stream(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        callback: &mut dyn FnMut(&str),
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);
        let messages = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect::<Vec<_>>();
        let mut body = json!({
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": true
        });
        if !tools.is_empty() {
            body["tools"] = json!(tools);
            body["tool_choice"] = json!("required");
        }
        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }
        let mut request = self.client.post(&self.base_url).json(&body);
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }
        let response = request.send().map_err(|e| {
            AgentError::Generation(format!("Failed to get response from LightLLM: {}", e))
        })?;

        use std::io::{BufRead, BufReader};
        let mut reader = BufReader::new(response);
        let mut content = String::new();
        let mut line = String::new();
        while reader.read_line(&mut line).map_err(|e| AgentError::Generation(e.to_string()))? > 0 {
            if line.starts_with("data: ") {
                let data = line.trim_start_matches("data: ").trim();
                if data == "[DONE]" {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(token) = val["choices"][0]["delta"]["content"].as_str() {
                        callback(token);
                        content.push_str(token);
                    }
                }
            }
            line.clear();
        }

        let response = OpenAIResponse {
            choices: vec![Choice {
                message: AssistantMessage {
                    role: MessageRole::Assistant,
                    content: Some(content),
                    tool_calls: None,
                    refusal: None,
                },
            }],
        };
        Ok(Box::new(response))
    }
}
