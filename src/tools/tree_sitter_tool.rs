use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{base::BaseTool, tool_traits::Tool};
use anyhow::Result;

use tree_sitter::{Parser};
use tree_sitter_rust as ts_rust;

#[derive(Deserialize, JsonSchema)]
#[schemars(title = "TreeSitterToolParams")]
pub struct TreeSitterToolParams {
    #[schemars(description = "The Rust source code to parse into an AST")]
    code: String,
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

    fn forward(&self, code: &str) -> Result<String> {
        let language = ts_rust::LANGUAGE;
        let mut parser = Parser::new();
        parser.set_language(&language.into())?;
        let tree = parser.parse(code, None).ok_or_else(|| anyhow::anyhow!("Failed to parse"))?;
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
        self.forward(&params.code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_sitter_tool() {
        let tool = TreeSitterTool::new();
        let params = TreeSitterToolParams { code: "fn main() {}".to_string() };
        let ast = <TreeSitterTool as Tool>::forward(&tool, params).unwrap();
        assert!(ast.contains("function_item"));
    }
}
