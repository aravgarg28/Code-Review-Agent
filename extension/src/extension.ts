import * as vscode from "vscode";
import { exec } from "child_process";
import { promisify } from "util";
import * as https from "https";
import * as http from "http";

const execAsync = promisify(exec);

interface PredictionResult {
  predicted_label: string;
  predicted_index: number;
  confidence: number;
  probabilities: Record<string, number>;
}

interface ReviewDiffResponse {
  prediction: PredictionResult;
  inference_time_ms: number;
}

async function getGitDiff(): Promise<string> {
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (!workspaceFolders || workspaceFolders.length === 0) {
    throw new Error("No workspace folder open");
  }

  const cwd = workspaceFolders[0].uri.fsPath;

  try {
    const { stdout } = await execAsync("git diff HEAD", { cwd });
    if (!stdout.trim()) {
      const { stdout: stagedDiff } = await execAsync("git diff --cached", {
        cwd,
      });
      if (!stagedDiff.trim()) {
        throw new Error(
          "No changes detected. Stage or modify files before reviewing."
        );
      }
      return stagedDiff;
    }
    return stdout;
  } catch (err: unknown) {
    if (err instanceof Error && err.message.includes("not a git repository")) {
      throw new Error("Current workspace is not a Git repository");
    }
    throw err;
  }
}

function postJSON(url: string, body: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const transport = parsed.protocol === "https:" ? https : http;

    const req = transport.request(
      {
        hostname: parsed.hostname,
        port: parsed.port,
        path: parsed.pathname,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          const data = Buffer.concat(chunks).toString();
          if (res.statusCode && res.statusCode >= 400) {
            reject(new Error(`API returned ${res.statusCode}: ${data}`));
          } else {
            resolve(data);
          }
        });
      }
    );

    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

async function reviewDiff(): Promise<void> {
  const config = vscode.workspace.getConfiguration("codeReviewAgent");
  const apiUrl: string = config.get("apiUrl", "http://localhost:8080");

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Code Review Agent",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Extracting git diff..." });

      let diff: string;
      try {
        diff = await getGitDiff();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        vscode.window.showErrorMessage(`Code Review Agent: ${msg}`);
        return;
      }

      progress.report({ message: "Running AI review..." });

      let response: ReviewDiffResponse;
      try {
        const raw = await postJSON(
          `${apiUrl}/api/v1/review/diff`,
          JSON.stringify({ diff })
        );
        response = JSON.parse(raw) as ReviewDiffResponse;
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        vscode.window.showErrorMessage(
          `Code Review Agent: API request failed — ${msg}`
        );
        return;
      }

      const { prediction, inference_time_ms } = response;

      if (prediction.predicted_label === "clean") {
        vscode.window.showInformationMessage(
          `Code Review Agent: No anti-patterns detected ` +
            `(${inference_time_ms.toFixed(0)}ms)`
        );
        return;
      }

      const probLines = Object.entries(prediction.probabilities)
        .sort(([, a], [, b]) => b - a)
        .map(([label, prob]) => `  ${label}: ${(prob * 100).toFixed(1)}%`)
        .join("\n");

      const panel = vscode.window.createOutputChannel("Code Review Agent");
      panel.clear();
      panel.appendLine(`Anti-pattern detected: ${prediction.predicted_label}`);
      panel.appendLine(
        `Confidence: ${(prediction.confidence * 100).toFixed(1)}%`
      );
      panel.appendLine(`Inference time: ${inference_time_ms.toFixed(0)}ms`);
      panel.appendLine("");
      panel.appendLine("Class probabilities:");
      panel.appendLine(probLines);
      panel.show();

      vscode.window.showWarningMessage(
        `Code Review Agent: [${prediction.predicted_label.toUpperCase()}] ` +
          `detected (${(prediction.confidence * 100).toFixed(1)}% confidence)`
      );
    }
  );
}

export function activate(context: vscode.ExtensionContext): void {
  const disposable = vscode.commands.registerCommand(
    "codeReviewAgent.reviewDiff",
    reviewDiff
  );
  context.subscriptions.push(disposable);
}

export function deactivate(): void {
  // no cleanup needed
}
