use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{generation::LogitsProcessor, models::llama::{Cache, Config, Llama, LlamaConfig, LlamaEosToks}};
use tokenizers::Tokenizer;

use crate::{errors::AgentError, models::model_traits::{Model, ModelResponse}, models::openai::ToolCall, models::types::{Message, MessageRole}, tools::ToolInfo};

pub struct CandleResponse {
    text: String,
}

impl ModelResponse for CandleResponse {
    fn get_response(&self) -> Result<String, AgentError> {
        Ok(self.text.clone())
    }

    fn get_tools_used(&self) -> Result<Vec<ToolCall>, AgentError> {
        Ok(vec![])
    }
}

#[derive(Clone, Debug)]
pub struct CandleModel {
    model: Llama,
    tokenizer: Tokenizer,
    config: Config,
    device: Device,
    temperature: f32,
}

impl CandleModel {
    pub fn new(model_dir: &str, temperature: Option<f32>) -> Result<Self> {
        let device = Device::Cpu;
        let config_path = format!("{}/config.json", model_dir);
        let llama_cfg: LlamaConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let config = llama_cfg.into_config(false);

        let weights = vec![format!("{}/model.safetensors", model_dir)];
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F16, &device)? };
        let model = Llama::load(vb, &config)?;
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            model,
            tokenizer,
            config,
            device,
            temperature: temperature.unwrap_or(0.7),
        })
    }

    fn generate(&self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let mut cache = Cache::new(true, DType::F16, &self.config, &self.device)?;
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let mut logits_processor = LogitsProcessor::new(299792458, Some(self.temperature as f64), None);
        let eos_id = match self.config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => Some(id),
            Some(LlamaEosToks::Multiple(ref ids)) => ids.first().cloned(),
            None => None,
        };

        for index in 0..max_new_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, tokens.len() - 1)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len() - context_size..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if let Some(eos) = eos_id {
                if next_token == eos {
                    break;
                }
            }
        }

        let text = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(text)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        callback: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let mut cache = Cache::new(true, DType::F16, &self.config, &self.device)?;
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let mut logits_processor =
            LogitsProcessor::new(299792458, Some(self.temperature as f64), None);
        let eos_id = match self.config.eos_token_id {
            Some(LlamaEosToks::Single(id)) => Some(id),
            Some(LlamaEosToks::Multiple(ref ids)) => ids.first().cloned(),
            None => None,
        };

        let mut output_tokens = Vec::new();
        for index in 0..max_new_tokens {
            let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
                (1, tokens.len() - 1)
            } else {
                (tokens.len(), 0)
            };
            let ctxt = &tokens[tokens.len() - context_size..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, context_index, &mut cache)?;
            let logits = logits.squeeze(0)?;
            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            output_tokens.push(next_token);
            let token_text = self
                .tokenizer
                .decode(&[next_token], false)
                .map_err(anyhow::Error::msg)?;
            callback(&token_text);
            if let Some(eos) = eos_id {
                if next_token == eos {
                    break;
                }
            }
        }

        let mut all_tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        all_tokens.extend(output_tokens);
        let text = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(anyhow::Error::msg)?;
        Ok(text)
    }
}

impl Model for CandleModel {
    fn run(
        &self,
        messages: Vec<Message>,
        _tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        _args: Option<HashMap<String, Vec<String>>>,
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let conversation = messages
            .iter()
            .map(|m| match m.role {
                MessageRole::User => format!("User: {}", m.content),
                MessageRole::Assistant => format!("Assistant: {}", m.content),
                MessageRole::System => format!("System: {}", m.content),
                MessageRole::ToolCall => format!("Tool: {}", m.content),
                MessageRole::ToolResponse => format!("ToolResponse: {}", m.content),
            })
            .collect::<Vec<_>>()
            .join("\n");

        let text = self
            .generate(&conversation, max_tokens.unwrap_or(256))
            .map_err(|e| AgentError::Generation(e.to_string()))?;
        Ok(Box::new(CandleResponse { text }))
    }

    fn run_stream(
        &self,
        messages: Vec<Message>,
        _tools: Vec<ToolInfo>,
        max_tokens: Option<usize>,
        _args: Option<HashMap<String, Vec<String>>>,
        callback: &mut dyn FnMut(&str),
    ) -> Result<Box<dyn ModelResponse>, AgentError> {
        let conversation = messages
            .iter()
            .map(|m| match m.role {
                MessageRole::User => format!("User: {}", m.content),
                MessageRole::Assistant => format!("Assistant: {}", m.content),
                MessageRole::System => format!("System: {}", m.content),
                MessageRole::ToolCall => format!("Tool: {}", m.content),
                MessageRole::ToolResponse => format!("ToolResponse: {}", m.content),
            })
            .collect::<Vec<_>>()
            .join("\n");

        let text = self
            .generate_stream(&conversation, max_tokens.unwrap_or(256), callback)
            .map_err(|e| AgentError::Generation(e.to_string()))?;
        Ok(Box::new(CandleResponse { text }))
    }
}

