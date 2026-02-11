"""
PARL Orchestrator — Code Review Example

Demonstrates how PARLOrchestrator handles parallel code review across
multiple files or modules, with each sub-agent reviewing a specific component.

This example shows:
- Decomposition of multi-file review into parallel sub-tasks
- Each sub-agent reviewing a specific file or module
- Structured review output (issues, severity, recommendations)
- Cross-file dependency analysis
- Synthesis of findings with prioritization
"""

import os
from swarms.structs.parl_orchestrator import PARLOrchestrator
from swarms.structs.swarm_router import SwarmRouter


# Sample code files to review (in a real scenario, these would be read from disk)
SAMPLE_FILES = {
    "user_service.py": '''
"""User authentication and management service."""
import hashlib
from typing import Optional

class UserService:
    def __init__(self):
        self.users = {}  # In-memory storage (not production-ready)

    def create_user(self, username: str, password: str) -> bool:
        # TODO: Add input validation
        # Store password as plain text (SECURITY ISSUE!)
        self.users[username] = password
        return True

    def authenticate(self, username: str, password: str) -> bool:
        if username not in self.users:
            return False
        return self.users[username] == password

    def get_user_data(self, username: str) -> Optional[dict]:
        # No authorization check (SECURITY ISSUE!)
        if username in self.users:
            return {"username": username, "password": self.users[username]}
        return None
''',

    "payment_processor.py": '''
"""Payment processing with external API."""
import requests

class PaymentProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.payment-provider.com/v1/charge"

    def process_payment(self, amount: float, card_number: str):
        # No error handling (RELIABILITY ISSUE!)
        # Logs sensitive data (SECURITY ISSUE!)
        print(f"Processing payment: {amount} from card {card_number}")

        response = requests.post(
            self.endpoint,
            json={
                "amount": amount,
                "card": card_number,
                "api_key": self.api_key  # API key in request body (SECURITY ISSUE!)
            },
            timeout=5  # Short timeout might cause issues
        )

        return response.json()

    def refund(self, transaction_id: str):
        # No idempotency handling (DATA ISSUE!)
        response = requests.post(
            f"{self.endpoint}/refund",
            json={"transaction_id": transaction_id}
        )
        return response.status_code == 200
''',

    "data_processor.py": '''
"""Batch data processing for analytics."""
import pandas as pd
from typing import List

def process_user_data(user_ids: List[int]) -> pd.DataFrame:
    # SQL injection vulnerability (SECURITY ISSUE!)
    query = f"SELECT * FROM users WHERE id IN ({','.join(map(str, user_ids))})"

    # No connection pooling (PERFORMANCE ISSUE!)
    conn = create_db_connection()
    df = pd.read_sql(query, conn)
    conn.close()

    # Memory inefficient for large datasets (PERFORMANCE ISSUE!)
    df['processed'] = df.apply(lambda row: expensive_operation(row), axis=1)

    return df

def expensive_operation(row):
    # Simulated expensive operation
    # No caching, runs for every row (PERFORMANCE ISSUE!)
    return row['value'] * 2

def create_db_connection():
    # Hardcoded credentials (SECURITY ISSUE!)
    return connect("postgresql://admin:password123@localhost/mydb")
''',
}


def main():
    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Example 1: Parallel code review with PARLOrchestrator
    print("=" * 80)
    print("Example 1: Parallel Multi-File Code Review")
    print("=" * 80)

    orchestrator = PARLOrchestrator(
        name="code-reviewer",
        description="Parallel code review orchestrator for security, performance, and quality analysis",
        orchestrator_model="gpt-4o-mini",
        sub_agent_model="gpt-4o-mini",
        max_parallel=5,          # Review multiple files in parallel
        max_iterations=1,        # Code review typically doesn't need gap-fill
        timeout=300,
        sub_agent_timeout=120,
    )

    # Build the review task with sample code
    files_summary = "\n\n".join(
        f"File: {filename}\n```python\n{code}\n```"
        for filename, code in SAMPLE_FILES.items()
    )

    task = f"""
    Perform a comprehensive code review of the following Python files.
    Focus on:
    - Security vulnerabilities (injection, auth, secrets management)
    - Performance issues (N+1 queries, memory leaks, inefficient algorithms)
    - Reliability concerns (error handling, edge cases, failure modes)
    - Code quality (maintainability, testing, documentation)

    For each issue found:
    - Severity: Critical/High/Medium/Low
    - Category: Security/Performance/Reliability/Quality
    - Description: What's wrong
    - Recommendation: How to fix it
    - Code snippet: Suggested fix if applicable

    Files to review:

    {files_summary}

    After individual file reviews, provide:
    - Cross-file dependency issues
    - Architecture-level concerns
    - Prioritized list of critical fixes
    """

    print("\nReviewing 3 Python files for security, performance, and quality issues...")
    print("Executing parallel code review... (this may take 1-2 minutes)\n")

    result = orchestrator.run(task)

    print("\n" + "=" * 80)
    print("Code Review Results:")
    print("=" * 80)
    print(result)

    # Example 2: Using SwarmRouter for code review
    print("\n\n" + "=" * 80)
    print("Example 2: Code Review via SwarmRouter")
    print("=" * 80)

    router = SwarmRouter(
        name="code-review-router",
        swarm_type="PARLOrchestrator",
    )

    # Simpler focused review
    task2 = """
    Review this authentication module for security issues:

    ```python
    import jwt
    from datetime import datetime, timedelta

    SECRET_KEY = "my-secret-key-123"  # Hardcoded secret

    def create_token(user_id: int) -> str:
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(days=30)  # Long expiration
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    def verify_token(token: str) -> dict:
        # No error handling for invalid tokens
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

    def admin_check(token: str) -> bool:
        data = verify_token(token)
        # Trusts user-provided claim without verification
        return data.get("is_admin", False)
    ```

    Identify security vulnerabilities and provide fixes.
    """

    print("\nTask:")
    print(task2[:200] + "..." if len(task2) > 200 else task2)
    print("\nExecuting security review...\n")

    result2 = router.run(task2)

    print("\n" + "=" * 80)
    print("Security Review Results:")
    print("=" * 80)
    print(result2)


if __name__ == "__main__":
    main()
