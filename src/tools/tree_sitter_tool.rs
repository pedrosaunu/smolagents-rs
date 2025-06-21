use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{base::BaseTool, tool_traits::Tool};
use anyhow::Result;

use tree_sitter::Parser;
use tree_sitter::{Language as TsLanguage};
use tree_sitter_rust as ts_rust;
use tree_sitter_python as ts_python;
use tree_sitter_javascript as ts_js;
use tree_sitter_bash as ts_bash;

#[derive(Deserialize, Serialize, JsonSchema, Clone)]
#[serde(rename_all = "lowercase")]
pub enum CodeLanguage {
    Rust,
    Python,
    Javascript,
    Bash,
}

fn default_language() -> CodeLanguage {
    CodeLanguage::Rust
}

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "TreeSitterToolParams")]
pub struct TreeSitterToolParams {
    #[schemars(description = "The source code to parse into an AST")]
    code: String,
    #[schemars(description = "The language of the source code (rust, python, javascript, bash)")]
    #[serde(default = "default_language")]
    language: CodeLanguage,
}

#[derive(Debug, Serialize, Clone)]
pub struct TreeSitterTool {
    pub tool: BaseTool,
}

impl TreeSitterTool {
    pub fn new() -> Self {
        TreeSitterTool {
            tool: BaseTool {
                name: "tree_sitter_parse",
                description: "Parse Rust code into an s-expression AST using tree-sitter.",
            },
        }
    }

    fn forward(&self, code: &str, language: &CodeLanguage) -> Result<String> {
        let lang: TsLanguage = match language {
            CodeLanguage::Rust => ts_rust::LANGUAGE.into(),
            CodeLanguage::Python => ts_python::LANGUAGE.into(),
            CodeLanguage::Javascript => ts_js::LANGUAGE.into(),
            CodeLanguage::Bash => ts_bash::LANGUAGE.into(),
        };
        let mut parser = Parser::new();
        parser.set_language(&lang)?;
        let tree = parser
            .parse(code, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse"))?;
        Ok(tree.root_node().to_sexp())
    }
}

impl Tool for TreeSitterTool {
    type Params = TreeSitterToolParams;

    fn name(&self) -> &'static str {
        self.tool.name
    }

    fn description(&self) -> &'static str {
        self.tool.description
    }

    fn forward(&self, params: TreeSitterToolParams) -> Result<String> {
        self.forward(&params.code, &params.language)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_sitter_tool() {
        let tool = TreeSitterTool::new();
        let params = TreeSitterToolParams {
            code: "fn main() {}".to_string(),
            language: CodeLanguage::Rust,
        };
        let ast = <TreeSitterTool as Tool>::forward(&tool, params).unwrap();
        assert!(ast.contains("function_item"));
    }

    #[test]
    fn test_tree_sitter_python() {
        let tool = TreeSitterTool::new();
        let params = TreeSitterToolParams {
            code: "def hello():\n  return 42".to_string(),
            language: CodeLanguage::Python,
        };
        let ast = <TreeSitterTool as Tool>::forward(&tool, params).unwrap();
        assert!(ast.contains("function_definition"));
    }
}
