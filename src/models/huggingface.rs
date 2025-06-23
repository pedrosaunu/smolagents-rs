use std::collections::HashMap;

use serde_json::json;

use crate::{
    errors::AgentError,
    models::model_traits::{Model, ModelResponse},
    models::openai::ToolCall,
    models::types::{Message, MessageRole},
    tools::ToolInfo,
};

#[derive(Debug)]
pub struct HuggingFaceResponse {
    text: String,
}

impl ModelResponse for HuggingFaceResponse {
    fn get_response(&self) -> Result<String, AgentError> {
        Ok(self.text.clone())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError> {
        Ok(vec![])
    }
}

#[derive(Debug, Clone)]
pub struct HuggingFaceModel {
    pub base_url: String,
    pub model_id: String,
    pub client: reqwest::blocking::Client,
    pub api_key: String,
    pub temperature: f32,
}

impl HuggingFaceModel {
    pub fn new(
        base_url: Option<&str>,
        model_id: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
    ) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("HF_API_KEY").expect("HF_API_KEY must be set")
        });
        let model_id = model_id.unwrap_or("HuggingFaceH4/zephyr-7b-beta").to_string();
        let base_url = base_url
            .unwrap_or("https://api-inference.huggingface.co/models")
            .to_string();
        let client = reqwest::blocking::Client::new();
        HuggingFaceModel {
            base_url,
            model_id,
            client,
            api_key,
            temperature: temperature.unwrap_or(0.5),
        }
    }
}

impl Model for HuggingFaceModel {
    fn run(
        &self,
        messages: Vec<Message>,
        _tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        _args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let conversation = messages
            .iter()
            .map(|m| format!("{}: {}", match m.role {
                MessageRole::User => "User",
                MessageRole::Assistant => "Assistant",
                MessageRole::System => "System",
                MessageRole::ToolCall => "Tool",
                MessageRole::ToolResponse => "ToolResponse",
            }, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let body = json!({
            "inputs": conversation,
            "parameters": {
                "max_new_tokens": max_tokens.unwrap_or(1500),
                "temperature": self.temperature
            }
        });

        let url = format!("{}/{}", self.base_url, self.model_id);
        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .map_err(|e| AgentError::Generation(format!("Failed to get response from Hugging Face: {}", e)))?;

        if response.status().is_success() {
            let value: serde_json::Value = response
                .json()
                .map_err(|e| AgentError::Generation(e.to_string()))?;
            let text = if let Some(arr) = value.as_array() {
                arr.first()
                    .and_then(|v| v.get("generated_text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string()
            } else {
                value
                    .get("generated_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string()
            };
            Ok(Box::new(HuggingFaceResponse { text }))
        } else {
            Err(AgentError::Generation(format!(
                "Failed to get response from Hugging Face: {}",
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
        let response = self.run(messages, tools, max_tokens, args)?;
        let text = response.get_response()?;
        for token in text.split_whitespace() {
            callback(token);
            callback(" ");
        }
        Ok(response)
    }
}

