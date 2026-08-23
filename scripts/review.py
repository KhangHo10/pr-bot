from google import genai
from pydantic import BaseModel, Field
from typing import List, Literal
import requests
import os

class Issue(BaseModel):
    file: str = Field(description="File path where the issue occurs.")
    description: str = Field(description="What the issue is and why it matters.")
    severity: Literal["low", "medium", "high"] = Field(description="How serious the issue is.")

class PRReview(BaseModel):
    summary: str = Field(description="A 1-2 sentence overview of the PR's changes.")
    issues: List[Issue] = Field(description="Specific problems found, if any.")
    suggestions: List[str] = Field(description="General improvement suggestions, non-blocking.")
    approve: bool = Field(description="Whether this PR looks safe to merge as-is.")

def build_prompt(diff: str) -> str:
    """Build the review prompt to send as the model's input"""
    return f"""You are reviewing a pull request. Analyze the diff below and identify:
    - Bugs or logic errors
    - Unclear or hand-to-maintain code
    - Missing edge case handling
    
    Be specific and reference file names where relevant. Keep the summary brief.
    If the PR looks clean, it's fine to return an empty issues list.
    
    Diff:
    {diff}
    """

def get_pr_diff(repo: str, pr_number: str, github_token: str) -> str:
    """Fetch the PR's diff from GitHub API"""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.diff", # let GitHub know what format to send back (default is JSON, but our is raw unified diff text b/c GitHub API support custom media types)
        "X-GitHub-Api-Version": "2022-11-28"     # Specify the API version to use (default is latest, but we want to be explicit so if GitHub decide to change the endpoint in future versions, we won't be affected)
    }    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text

def review_diff(diff: str, api_key: str) -> str:
    """Send the MR diff to Gemini and get back review feedback."""
    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model="gemini-3.7-flash",
        input=build_prompt(diff),
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": PRReview.model_json_schema()
        },
    )

    response = PRReview.model_validate_json(interaction.output_text)
    return response

def format_review(review: PRReview) -> str:
    """Convert a PRReview object into a markdown comment string"""
    lines = [f"### 🤖 PR Review\n", f"{review.summary}\n"]

    if review.issues:
        lines.append("**Issue found: **\n")

        for issue in review.issues:
            lines.append(f"- **[{issue.severity.upper()}]** `{issue.file}`: {issue.description}")
            lines.append("")
    else:
        lines.append("No issues found.\n")

    if review.suggestions:
        lines.append("**Suggestions:**\n")

        for suggestion in review.suggestions:
            lines.append(f"- {suggestion}")
        lines.append("")

    verdict = "✅ Looks good to merge" if review.approve else "⚠️ Changes recommended before merging"
    lines.append(f"\n{verdict}")

    return "\n".join(lines)

def post_comment(repo: str, pr_number: str, github_token: str, comment: str) -> str:
    """Post the review as a comment on the PR"""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    response = requests.post(url, headers=headers, json={"body": comment})
    response.raise_for_status()

def main():
    """Orchestrates the three steps above using evn vars from the workflow"""
    repo = os.environ["REPO"]
    pr_number = os.environ["PR_NUMBER"]
    github_token = os.environ["GITHUB_TOKEN"]
    api_key = os.environ["GEMINI_API_KEY"]

    diff = get_pr_diff(repo, pr_number, github_token)
    review_object = review_diff(diff, api_key)
    review_format = format_review(review_object)
    post_comment(repo, pr_number, github_token, review_format)

if __name__ == "__main__":
    main()