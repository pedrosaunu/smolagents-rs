use std::collections::HashMap;

use crate::errors::AgentError;
use crate::models::model_traits::{Model, ModelResponse};
use crate::models::openai::{AssistantMessage, Choice, OpenAIResponse};
use crate::models::types::{Message, MessageRole};
use crate::tools::ToolInfo;
use anyhow::Result;
use reqwest::blocking::Client;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct AzureOpenAIModel {
    pub base_url: String,
    pub deployment_id: String,
    pub api_version: String,
    pub client: Client,
    pub temperature: f32,
    pub api_key: String,
}

impl AzureOpenAIModel {
    pub fn new(
        endpoint: Option<&str>,
        deployment_id: Option<&str>,
        api_version: Option<&str>,
        temperature: Option<f32>,
        api_key: Option<String>,
    ) -> Self {
        let api_key = api_key.unwrap_or_else(|| {
            std::env::var("AZURE_OPENAI_API_KEY").expect("AZURE_OPENAI_API_KEY must be set")
        });
        let endpoint: String = endpoint.map(|s| s.to_string()).unwrap_or_else(|| {
            std::env::var("AZURE_OPENAI_ENDPOINT").expect("AZURE_OPENAI_ENDPOINT must be set")
        });
        let deployment_id: String = deployment_id.map(|s| s.to_string()).unwrap_or_else(|| {
            std::env::var("AZURE_OPENAI_DEPLOYMENT_ID")
                .expect("AZURE_OPENAI_DEPLOYMENT_ID must be set")
        });
        let api_version: String = api_version.map(|s| s.to_string()).unwrap_or_else(|| {
            std::env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2024-02-15-preview".to_string())
        });
        let base_url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            endpoint, deployment_id, api_version
        );
        let client = Client::new();
        Self {
            base_url,
            deployment_id,
            api_version,
            client,
            temperature: temperature.unwrap_or(0.5),
            api_key,
        }
    }
}

impl Model for AzureOpenAIModel {
    fn run(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);

        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();
        let mut body = json!({
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        });

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
        }

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }

        let response = self
            .client
            .post(&self.base_url)
            .header("api-key", &self.api_key)
            .json(&body)
            .send()
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from Azure OpenAI: {}", e))
            })?;

        match response.status() {
            reqwest::StatusCode::OK => {
                let response = response.json::<OpenAIResponse>().unwrap();
                Ok(Box::new(response))
            }
            _ => Err(AgentError::Generation(format!(
                "Failed to get response from Azure OpenAI: {}",
                response.text().unwrap()
            ))),
        }
    }

    fn run_stream(
        &self,
        messages: Vec<Message>,
        tools_to_call_from: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        args: Option<HashMap<String, Vec<String>>>,
        callback: &mut dyn FnMut(&str),
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let max_tokens = max_tokens.unwrap_or(1500);

        let messages = messages
            .iter()
            .map(|message| {
                json!({
                    "role": message.role,
                    "content": message.content
                })
            })
            .collect::<Vec<_>>();
        let mut body = json!({
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": true
        });

        if !tools_to_call_from.is_empty() {
            body["tools"] = json!(tools_to_call_from);
            body["tool_choice"] = json!("required");
        }

        if let Some(args) = args {
            let body_map = body.as_object_mut().unwrap();
            for (key, value) in args {
                body_map.insert(key, json!(value));
            }
        }

        let response = self
            .client
            .post(&self.base_url)
            .header("api-key", &self.api_key)
            .json(&body)
            .send()
            .map_err(|e| {
                AgentError::Generation(format!("Failed to get response from Azure OpenAI: {}", e))
            })?;

        use std::io::{BufRead, BufReader};

        let mut reader = BufReader::new(response);
        let mut content = String::new();
        let mut line = String::new();
        while reader
            .read_line(&mut line)
            .map_err(|e| AgentError::Generation(e.to_string()))?
            > 0
        {
            if line.starts_with("data: ") {
                let data = line.trim_start_matches("data: ").trim();
                if data == "[DONE]" {
                    break;
                }
                if let Ok(val) = serde_json::from_str::<Value>(data) {
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
