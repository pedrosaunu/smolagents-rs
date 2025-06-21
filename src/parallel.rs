use std::sync::Arc;
use std::thread;

use anyhow::Result;

use crate::agents::Agent;

/// Run multiple tasks in parallel using a fresh agent instance for each task.
///
/// # Arguments
///
/// * `builder` - An `Arc` containing a closure that can create a new agent.
/// * `tasks` - Slice of task strings to be executed.
///
/// # Returns
///
/// A vector containing the result of each task in the same order as provided.
pub fn run_tasks_parallel<A>(
    builder: Arc<dyn Fn() -> A + Send + Sync>,
    tasks: &[String],
) -> Vec<Result<String>>
where
    A: Agent + 'static,
{
    let mut handles = Vec::new();

    for task in tasks.iter().cloned() {
        let builder = builder.clone();
        handles.push(thread::spawn(move || {
            let mut agent = builder();
            agent.run(&task, false, true)
        }));
    }

    handles
        .into_iter()
        .map(|h| h.join().unwrap_or_else(|_| Err(anyhow::anyhow!("Thread panicked"))))
        .collect()
}
