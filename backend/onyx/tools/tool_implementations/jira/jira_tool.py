import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, validator
from enum import Enum

from jira import JIRA, Issue
from jira.exceptions import JIRAError

from onyx.tools.base_tool import BaseTool
from onyx.tools.models import ToolResponse
from onyx.connectors.jira.utils import build_jira_client, build_jira_url
from onyx.utils.logger import setup_logger

if TYPE_CHECKING:
    from onyx.llm.interfaces import LLM
    from onyx.llm.models import PreviousMessage

logger = setup_logger()

JIRA_STRUCTURED_RESPONSE_ID = "jira_structured"

# Constants for limiting results
DEFAULT_MAX_COMMENTS = 5
DEFAULT_MAX_WORKLOGS = 10
DEFAULT_MAX_CHANGES = 5

class JiraActionType(str, Enum):
    COUNT_OPEN_TICKETS = "count_open_tickets"
    LATEST_TICKET_IN_EPIC = "latest_ticket_in_epic"
    SEARCH_JQL = "search_jql"
    GET_ISSUE_DETAILS = "get_issue_details"
    LIST_EPICS = "list_epics"
    GET_SPRINT_INFO = "get_sprint_info"
    TICKETS_BY_ASSIGNEE = "tickets_by_assignee"
    OVERDUE_TICKETS = "overdue_tickets"
    TICKETS_BY_STATUS = "tickets_by_status"
    WORKLOG_SUMMARY = "worklog_summary"
    BLOCKED_TICKETS = "blocked_tickets"
    PRIORITY_BREAKDOWN = "priority_breakdown"

class TimeFrame(str, Enum):
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"

class JiraAction(BaseModel):
    action: JiraActionType = Field(description="The action to perform on Jira")
    project_key: Optional[str] = Field(None, description="Project key (e.g., 'ABC')")
    epic_key: Optional[str] = Field(None, description="Epic key (e.g., 'ABC-123')")
    issue_key: Optional[str] = Field(None, description="Issue key for specific operations")
    assignee: Optional[str] = Field(None, description="Assignee username or display name")
    status: Optional[str] = Field(None, description="Issue status")
    priority: Optional[str] = Field(None, description="Issue priority")
    extra_jql: Optional[str] = Field(None, description="Additional JQL filters")
    jql: Optional[str] = Field(None, description="Full JQL query for search_jql action")
    time_frame: Optional[TimeFrame] = Field(None, description="Time frame for filtering")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    include_subtasks: bool = Field(False, description="Include subtasks in results")
    include_comments: bool = Field(False, description="Include comments in detailed results")
    include_worklog: bool = Field(False, description="Include worklog information")
    group_by: Optional[str] = Field(None, description="Group results by field (assignee, status, priority)")

    @validator('time_frame')
    def validate_time_frame(cls, v, values):
        if v == TimeFrame.CUSTOM and not (values.get('start_date') or values.get('end_date')):
            raise ValueError("start_date and end_date required for custom time_frame")
        return v

class JiraStructuredTool(BaseTool):
    _NAME = "jira_structured_query"
    _DESCRIPTION = (
        "Advanced Jira query tool supporting multiple operations: ticket counts, epic analysis, "
        "sprint information, worklog summaries, blocked tickets, priority breakdowns, and more. "
        "Supports flexible filtering by assignee, status, priority, time frames, and custom JQL."
    )
    _DISPLAY_NAME = "Advanced Jira Query Tool"

    def __init__(
            self,
            jira_base_url: str,
            credentials: dict[str, Any],
            tool_id: int,
            request_timeout_sec: int = 30,
            cache_duration_sec: int = 300,  # 5 minutes cache
            custom_field_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self._id = tool_id
        self.jira: JIRA = build_jira_client(credentials=credentials, jira_base=jira_base_url.rstrip("/"))
        self.request_timeout_sec = request_timeout_sec
        self.cache_duration_sec = cache_duration_sec
        self._cache: Dict[str, tuple[datetime, Any]] = {}
        
        # Default custom field mapping - can be overridden
        self.custom_fields = custom_field_mapping or {
            'epic_link': 'customfield_10014',
            'story_points': 'customfield_10016',
            'sprint': 'customfield_10020'
        }

    def _get_cached_or_fetch(self, cache_key: str, fetch_fn):
        """Simple in-memory caching to reduce API calls"""
        now = datetime.now()
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (now - cached_time).total_seconds() < self.cache_duration_sec:
                return cached_data

        result = fetch_fn()
        self._cache[cache_key] = (now, result)
        return result

    def _build_time_filter(self, time_frame: Optional[TimeFrame], start_date: Optional[str], end_date: Optional[str]) -> str:
        """Build JQL time filter based on time frame"""
        if not time_frame:
            return ""

        now = datetime.now()
        if time_frame == TimeFrame.TODAY:
            return f"updated >= -{1}d"
        elif time_frame == TimeFrame.WEEK:
            return f"updated >= -{7}d"
        elif time_frame == TimeFrame.MONTH:
            return f"updated >= -{30}d"
        elif time_frame == TimeFrame.QUARTER:
            return f"updated >= -{90}d"
        elif time_frame == TimeFrame.YEAR:
            return f"updated >= -{365}d"
        elif time_frame == TimeFrame.CUSTOM:
            filters = []
            if start_date:
                filters.append(f"updated >= '{start_date}'")
            if end_date:
                filters.append(f"updated <= '{end_date}'")
            return " AND ".join(filters)
        return ""

    def _sanitize_jql_value(self, value: str) -> str:
        """Sanitize JQL values to prevent injection"""
        if not value:
            return value
        # Escape double quotes and backslashes
        return value.replace('\\', '\\\\').replace('"', '\\"')

    def _build_base_jql(self, params: JiraAction) -> List[str]:
        """Build base JQL filters from parameters"""
        filters = []

        if params.project_key:
            sanitized_project = self._sanitize_jql_value(params.project_key)
            filters.append(f'project = "{sanitized_project}"')

        if params.epic_key:
            # Epic keys don't need quotes in JQL
            filters.append(f"(parent = {params.epic_key} OR 'Epic Link' = {params.epic_key})")

        if params.assignee:
            sanitized_assignee = self._sanitize_jql_value(params.assignee)
            filters.append(f'assignee = "{sanitized_assignee}"')

        if params.status:
            sanitized_status = self._sanitize_jql_value(params.status)
            filters.append(f'status = "{sanitized_status}"')

        if params.priority:
            sanitized_priority = self._sanitize_jql_value(params.priority)
            filters.append(f'priority = "{sanitized_priority}"')

        time_filter = self._build_time_filter(params.time_frame, params.start_date, params.end_date)
        if time_filter:
            filters.append(time_filter)

        if params.extra_jql:
            # Note: extra_jql is not sanitized as it's meant to be raw JQL
            # Users should be careful with this parameter
            filters.append(f"({params.extra_jql})")

        return filters

    def _extract_issue_data(self, issue: Issue, include_comments: bool = False, include_worklog: bool = False) -> Dict[str, Any]:
        """Extract comprehensive issue data"""
        data = {
            "key": issue.key,
            "summary": getattr(issue.fields, "summary", None),
            "status": getattr(getattr(issue.fields, "status", None), "name", None),
            "assignee": getattr(getattr(issue.fields, "assignee", None), "displayName", None),
            "reporter": getattr(getattr(issue.fields, "reporter", None), "displayName", None),
            "priority": getattr(getattr(issue.fields, "priority", None), "name", None),
            "issuetype": getattr(getattr(issue.fields, "issuetype", None), "name", None),
            "created": getattr(issue.fields, "created", None),
            "updated": getattr(issue.fields, "updated", None),
            "duedate": getattr(issue.fields, "duedate", None),
            "labels": getattr(issue.fields, "labels", []),
            "components": [c.name for c in getattr(issue.fields, "components", [])],
        }

        # Build URL safely
        try:
            data["url"] = build_jira_url(self.jira, issue.key)
        except Exception:
            # Fallback URL construction
            base_url = str(self.jira._options['server']).rstrip('/')
            data["url"] = f"{base_url}/browse/{issue.key}"

        # Add epic information if available
        epic_field = self.custom_fields.get('epic_link', 'customfield_10014')
        epic_link = getattr(issue.fields, epic_field, None) or getattr(issue.fields, "parent", None)
        if epic_link:
            data["epic"] = getattr(epic_link, "key", str(epic_link))

        # Add story points if available
        story_points_field = self.custom_fields.get('story_points', 'customfield_10016')
        story_points = getattr(issue.fields, story_points_field, None)
        if story_points:
            data["story_points"] = story_points

        if include_comments:
            comments = getattr(issue.fields, "comment", None)
            if comments and hasattr(comments, "comments"):
                data["comments"] = [
                    {
                        "author": c.author.displayName,
                        "body": c.body,
                        "created": c.created
                    } for c in comments.comments[-DEFAULT_MAX_COMMENTS:]  # Last N comments
                ]

        if include_worklog:
            try:
                worklogs = self.jira.worklogs(issue.key)
                data["worklog"] = [
                    {
                        "author": w.author.displayName,
                        "timeSpent": w.timeSpent,
                        "started": w.started,
                        "comment": getattr(w, "comment", "")
                    } for w in worklogs[-DEFAULT_MAX_WORKLOGS:]  # Last N worklogs
                ]
                data["total_time_spent"] = sum(int(w.timeSpentSeconds or 0) for w in worklogs)
            except Exception as e:
                logger.warning(f"Could not fetch worklog for {issue.key}: {e}")

        return data

    def _count_open_tickets(self, params: JiraAction) -> Dict[str, Any]:
        """Count open tickets with advanced filtering"""
        # Simple approach - build filters step by step
        filters = []

        # Base filter for open tickets
        if params.status:
            sanitized_status = self._sanitize_jql_value(params.status)
            filters.append(f'status = "{sanitized_status}"')
        else:
            filters.append("(statusCategory = 'To Do' OR statusCategory = 'In Progress')")

        # Add specific filters
        if params.project_key:
            sanitized_project = self._sanitize_jql_value(params.project_key)
            filters.append(f'project = "{sanitized_project}"')

        if params.epic_key:
            filters.append(f"(parent = {params.epic_key} OR 'Epic Link' = {params.epic_key})")

        if params.assignee:
            sanitized_assignee = self._sanitize_jql_value(params.assignee)
            filters.append(f'assignee = "{sanitized_assignee}"')

        if params.priority:
            sanitized_priority = self._sanitize_jql_value(params.priority)
            filters.append(f'priority = "{sanitized_priority}"')

        # Time filter
        time_filter = self._build_time_filter(params.time_frame, params.start_date, params.end_date)
        if time_filter:
            filters.append(time_filter)

        if params.extra_jql:
            filters.append(f"({params.extra_jql})")

        if not params.include_subtasks:
            filters.append("issuetype != Sub-task")

        jql = " AND ".join(filters)

        try:
            def fetch_count():
                result = self.jira.search_issues(jql_str=jql, startAt=0, maxResults=0, fields="none")
                return result.total

            cache_key = f"count_{hash(jql)}"
            count = self._get_cached_or_fetch(cache_key, fetch_count)

            return {
                "type": "count_open_tickets",
                "jql": jql,
                "count": count,
                "filters_applied": {
                    "project": params.project_key,
                    "epic": params.epic_key,
                    "assignee": params.assignee,
                    "status": params.status,
                    "time_frame": params.time_frame,
                    "include_subtasks": params.include_subtasks
                }
            }
        except JIRAError as e:
            return {
                "type": "count_open_tickets",
                "error": f"Count query failed: {e}",
                "jql": jql
            }

    def _get_issue_details(self, issue_key: str, params: JiraAction) -> Dict[str, Any]:
        """Get detailed information about a specific issue"""
        try:
            logger.info(f"Fetching details for issue: {issue_key}")
            issue = self.jira.issue(issue_key, expand='changelog')
            data = self._extract_issue_data(issue, params.include_comments, params.include_worklog)

            # Add change history
            if hasattr(issue, 'changelog'):
                data["recent_changes"] = [
                    {
                        "created": h.created,
                        "author": h.author.displayName,
                        "items": [
                            {
                                "field": item.field,
                                "from": item.fromString,
                                "to": item.toString
                            } for item in h.items
                        ]
                    } for h in issue.changelog.histories[-DEFAULT_MAX_CHANGES:]  # Last N changes
                ]

            return {"type": "issue_details", "issue": data}
        except JIRAError as e:
            logger.error(f"Failed to fetch issue {issue_key}: {e}")
            return {
                "type": "issue_details", 
                "error": f"Issue {issue_key} not found or inaccessible: {e}",
                "issue_key": issue_key,
                "details": str(e)
            }

    def _list_epics(self, params: JiraAction) -> Dict[str, Any]:
        """List epics in project with their progress"""
        filters = ['issuetype = Epic']
        filters.extend(self._build_base_jql(params))
        jql = " AND ".join(filters)

        issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results)
        epics = []

        for epic in issues:
            epic_data = self._extract_issue_data(epic)

            # Get epic progress
            epic_jql = f"'Epic Link' = {epic.key}"
            epic_issues = self.jira.search_issues(jql_str=epic_jql, maxResults=0, fields="status")

            epic_data.update({
                "total_issues": epic_issues.total,
                "progress": f"{epic_issues.total} issues"
            })
            epics.append(epic_data)

        return {"type": "list_epics", "jql": jql, "epics": epics}

    def _tickets_by_assignee(self, params: JiraAction) -> Dict[str, Any]:
        """Get tickets grouped by assignee"""
        filters = self._build_base_jql(params)
        if not any("statusCategory" in f for f in filters):
            filters.insert(0, "(statusCategory = 'To Do' OR statusCategory = 'In Progress')")

        jql = " AND ".join(filters)
        issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results)

        by_assignee = {}
        for issue in issues:
            assignee = getattr(getattr(issue.fields, "assignee", None), "displayName", "Unassigned")
            if assignee not in by_assignee:
                by_assignee[assignee] = []
            by_assignee[assignee].append(self._extract_issue_data(issue))

        return {
            "type": "tickets_by_assignee",
            "jql": jql,
            "summary": {assignee: len(tickets) for assignee, tickets in by_assignee.items()},
            "details": by_assignee
        }

    def _overdue_tickets(self, params: JiraAction) -> Dict[str, Any]:
        """Find overdue tickets"""
        filters = [
            "duedate < now()",
            "(statusCategory = 'To Do' OR statusCategory = 'In Progress')"
        ]
        filters.extend(self._build_base_jql(params))
        jql = " AND ".join(filters)

        issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results)
        overdue_tickets = [self._extract_issue_data(issue) for issue in issues]

        return {
            "type": "overdue_tickets",
            "jql": jql,
            "count": len(overdue_tickets),
            "tickets": overdue_tickets
        }

    def _priority_breakdown(self, params: JiraAction) -> Dict[str, Any]:
        """Get breakdown of tickets by priority"""
        filters = self._build_base_jql(params)
        if not any("statusCategory" in f for f in filters):
            filters.insert(0, "(statusCategory = 'To Do' OR statusCategory = 'In Progress')")

        jql = " AND ".join(filters)
        issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results, fields="priority,key,summary")

        by_priority = {}
        for issue in issues:
            priority = getattr(getattr(issue.fields, "priority", None), "name", "No Priority")
            if priority not in by_priority:
                by_priority[priority] = 0
            by_priority[priority] += 1

        return {
            "type": "priority_breakdown",
            "jql": jql,
            "breakdown": by_priority,
            "total": sum(by_priority.values())
        }

    def _blocked_tickets(self, params: JiraAction) -> Dict[str, Any]:
        """Find blocked tickets"""
        filters = [
            "(status = 'Blocked' OR labels in ('blocked', 'Blocked') OR flagged = 'Impediment')"
        ]
        filters.extend(self._build_base_jql(params))
        jql = " AND ".join(filters)

        issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results)
        blocked_tickets = [self._extract_issue_data(issue) for issue in issues]

        return {
            "type": "blocked_tickets",
            "jql": jql,
            "count": len(blocked_tickets),
            "tickets": blocked_tickets
        }

    def _get_sprint_info(self, params: JiraAction) -> Dict[str, Any]:
        """Get sprint information"""
        try:
            filters = self._build_base_jql(params)
            if not filters:
                filters = ["sprint is not EMPTY"]
            else:
                filters.append("sprint is not EMPTY")
            
            jql = " AND ".join(filters)
            issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results, fields="sprint,key,summary,status")
            
            sprint_info = {}
            for issue in issues:
                sprints = getattr(issue.fields, "sprint", []) or []
                if not isinstance(sprints, list):
                    sprints = [sprints]
                
                for sprint in sprints:
                    if sprint:
                        sprint_name = getattr(sprint, "name", str(sprint))
                        if sprint_name not in sprint_info:
                            sprint_info[sprint_name] = []
                        sprint_info[sprint_name].append(self._extract_issue_data(issue))
            
            return {
                "type": "sprint_info",
                "jql": jql,
                "sprints": sprint_info,
                "total_sprints": len(sprint_info)
            }
        except Exception as e:
            return {
                "type": "sprint_info",
                "error": f"Failed to get sprint info: {e}",
                "jql": jql if 'jql' in locals() else "N/A"
            }

    def _tickets_by_status(self, params: JiraAction) -> Dict[str, Any]:
        """Get tickets grouped by status"""
        filters = self._build_base_jql(params)
        jql = " AND ".join(filters) if filters else "project is not EMPTY"
        
        try:
            issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results, fields="status,key,summary")
            
            by_status = {}
            for issue in issues:
                status = getattr(getattr(issue.fields, "status", None), "name", "Unknown")
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(self._extract_issue_data(issue))
            
            return {
                "type": "tickets_by_status",
                "jql": jql,
                "summary": {status: len(tickets) for status, tickets in by_status.items()},
                "details": by_status
            }
        except Exception as e:
            return {
                "type": "tickets_by_status",
                "error": f"Failed to get tickets by status: {e}",
                "jql": jql
            }

    def _worklog_summary(self, params: JiraAction) -> Dict[str, Any]:
        """Get worklog summary"""
        filters = self._build_base_jql(params)
        jql = " AND ".join(filters) if filters else "worklogDate >= -30d"
        
        try:
            issues = self.jira.search_issues(jql_str=jql, maxResults=params.max_results)
            
            total_time = 0
            worklog_data = []
            
            for issue in issues:
                try:
                    worklogs = self.jira.worklogs(issue.key)
                    issue_time = sum(int(w.timeSpentSeconds or 0) for w in worklogs)
                    total_time += issue_time
                    
                    if worklogs:
                        worklog_data.append({
                            "issue": issue.key,
                            "summary": getattr(issue.fields, "summary", ""),
                            "time_spent_seconds": issue_time,
                            "time_spent_hours": round(issue_time / 3600, 2),
                            "worklog_count": len(worklogs)
                        })
                except Exception as e:
                    logger.warning(f"Could not fetch worklog for {issue.key}: {e}")
            
            return {
                "type": "worklog_summary",
                "jql": jql,
                "total_time_seconds": total_time,
                "total_time_hours": round(total_time / 3600, 2),
                "issues_with_worklog": len(worklog_data),
                "worklog_details": worklog_data
            }
        except Exception as e:
            return {
                "type": "worklog_summary",
                "error": f"Failed to get worklog summary: {e}",
                "jql": jql
            }

    def get_args_for_non_tool_calling_llm(
            self,
            query: str,
            history: "list[PreviousMessage]",
            llm: "LLM",
            force_run: bool = False,
    ) -> dict[str, Any] | None:
        """Enhanced intent detection for natural language queries"""
        if not query:
            return None

        lower_q = query.lower()

        # Enhanced pattern matching with regex
        patterns = {
            # Count patterns
            r"\b(?:how many|count|number of).*(?:open|active|unresolved|pending)\b": {"action": "count_open_tickets"},
            r"\b(?:open|active|unresolved).*(?:tickets|issues|tasks).*(?:count|number)\b": {"action": "count_open_tickets"},

            # Latest/recent patterns
            r"\b(?:latest|recent|last).*(?:ticket|issue).*(?:in|for).*epic\b": {"action": "latest_ticket_in_epic"},

            # Epic patterns
            r"\b(?:list|show|get).*epics?\b": {"action": "list_epics"},

            # Assignee patterns
            r"\b(?:tickets?|issues?).*(?:assigned to|by assignee|per person)\b": {"action": "tickets_by_assignee"},

            # Overdue patterns
            r"\b(?:overdue|past due|late).*(?:tickets?|issues?)\b": {"action": "overdue_tickets"},

            # Priority patterns
            r"\b(?:priority|priorities).*(?:breakdown|distribution|summary)\b": {"action": "priority_breakdown"},

            # Blocked patterns
            r"\b(?:blocked|stuck|impediment).*(?:tickets?|issues?)\b": {"action": "blocked_tickets"},

            # Details patterns - more specific patterns first
            r"\b(?:detail|details|info|information|show|get|fetch).*?(?:about|for|of|the)?\s*(?:ticket|issue)?\s*([A-Z]{2,10}[-‑]\d+)\b": {"action": "get_issue_details"},
            r"\b(?:what|tell me).*?(?:about|is)\s*(?:ticket|issue)?\s*([A-Z]{2,10}[-‑]\d+)\b": {"action": "get_issue_details"},

            # Direct issue key patterns - handle both regular hyphen and en-dash
            r"\b([A-Z]{2,10}[-‑]\d+)\b": {"action": "get_issue_details"},
            r"^key\s*=\s*([A-Z]+[-‑]\d+)$": {"action": "search_jql"},
        }

        # Extract entities
        project_match = re.search(r'\bproject\s+([A-Z]{2,10})\b', query, re.IGNORECASE)
        issue_match = re.search(r'\b([A-Z]{2,10}[-‑]\d+)\b', query)
        assignee_match = re.search(r'\b(?:assigned to|assignee)\s+([a-zA-Z0-9._-]+)\b', query, re.IGNORECASE)
        time_matches = {
            'today': r'\btoday\b',
            'week': r'\b(?:this week|past week|last week)\b',
            'month': r'\b(?:this month|past month|last month)\b'
        }

        result = None

        # Try pattern matching
        for pattern, base_result in patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                result = base_result.copy()
                break

        # If force_run and no pattern matched, default to JQL search
        if not result and force_run:
            result = {"action": "search_jql", "jql": query}

        if result:
            # Add extracted entities
            if project_match:
                result["project_key"] = project_match.group(1).upper()
            if issue_match:
                # Normalize the issue key by replacing en-dash with regular hyphen
                normalized_issue_key = issue_match.group(1).upper().replace('‑', '-')
                if result["action"] == "get_issue_details":
                    result["issue_key"] = normalized_issue_key
                elif result["action"] == "latest_ticket_in_epic":
                    result["epic_key"] = normalized_issue_key
                elif result["action"] == "search_jql":
                    # For JQL search, construct the query
                    result["jql"] = f"key = {normalized_issue_key}"
            if assignee_match:
                result["assignee"] = assignee_match.group(1)

            # Add time frame
            for time_frame, pattern in time_matches.items():
                if re.search(pattern, lower_q):
                    result["time_frame"] = time_frame
                    break

            # Handle direct JQL queries
            if result["action"] == "search_jql" and "jql" not in result:
                # Check if the query looks like a JQL query
                if re.search(r'\b(?:key|project|assignee|status|priority)\s*[=!<>]', query, re.IGNORECASE):
                    result["jql"] = query

        return result

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": [action.value for action in JiraActionType],
                            "description": "The action to perform. Available actions:\n"
                                         "- get_issue_details: Get detailed info about a specific issue (requires issue_key)\n"
                                         "- search_jql: Search using JQL query (requires jql)\n"
                                         "- count_open_tickets: Count open tickets with filters\n"
                                         "- list_epics: List epics in project\n"
                                         "- latest_ticket_in_epic: Get latest ticket in epic (requires epic_key)\n"
                                         "- tickets_by_assignee: Group tickets by assignee\n"
                                         "- overdue_tickets: Find overdue tickets\n"
                                         "- tickets_by_status: Group tickets by status\n"
                                         "- worklog_summary: Get worklog summary\n"
                                         "- blocked_tickets: Find blocked tickets\n"
                                         "- priority_breakdown: Get priority breakdown\n"
                                         "- get_sprint_info: Get sprint information"
                        },
                        "issue_key": {
                            "type": "string", 
                            "description": "Issue key (e.g., 'TRAVO-23767') - required for get_issue_details action"
                        },
                        "jql": {
                            "type": "string", 
                            "description": "Full JQL query - required for search_jql action. Example: 'key = TRAVO-23767'"
                        },
                        "epic_key": {
                            "type": "string", 
                            "description": "Epic key (e.g., 'TRAVO-123') - required for latest_ticket_in_epic action"
                        },
                        "project_key": {"type": "string", "description": "Project key (e.g., 'TRAVO')"},
                        "assignee": {"type": "string", "description": "Assignee username or display name"},
                        "status": {"type": "string", "description": "Issue status (e.g., 'In Progress', 'Done')"},
                        "priority": {"type": "string", "description": "Issue priority (e.g., 'High', 'Medium', 'Low')"},
                        "extra_jql": {"type": "string", "description": "Additional JQL filters to combine with other parameters"},
                        "time_frame": {
                            "type": "string",
                            "enum": [tf.value for tf in TimeFrame],
                            "description": "Time frame for filtering: today, week, month, quarter, year, custom"
                        },
                        "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD) - used with time_frame=custom"},
                        "end_date": {"type": "string", "description": "End date (YYYY-MM-DD) - used with time_frame=custom"},
                        "max_results": {
                            "type": "integer", 
                            "minimum": 1, 
                            "maximum": 100,
                            "default": 20,
                            "description": "Maximum number of results to return (default: 20)"
                        },
                        "include_subtasks": {
                            "type": "boolean", 
                            "default": False,
                            "description": "Include subtasks in results (default: false)"
                        },
                        "include_comments": {
                            "type": "boolean", 
                            "default": False,
                            "description": "Include comments in detailed results (default: false)"
                        },
                        "include_worklog": {
                            "type": "boolean", 
                            "default": False,
                            "description": "Include worklog information (default: false)"
                        },
                        "group_by": {"type": "string", "description": "Group results by field (assignee, status, priority)"}
                    },
                    "required": ["action"],
                },
            },
        }

    def run(self, override_kwargs: None = None, **llm_kwargs: Any) -> Generator[ToolResponse, None, None]:
        try:
            logger.info(f"Jira tool called with parameters: {llm_kwargs}")
            params = JiraAction(**llm_kwargs)

            action_map = {
                JiraActionType.COUNT_OPEN_TICKETS: self._count_open_tickets,
                JiraActionType.LATEST_TICKET_IN_EPIC: lambda p: self._latest_ticket_in_epic(p.epic_key),
                JiraActionType.SEARCH_JQL: lambda p: self._search_jql(p.jql, p.max_results),
                JiraActionType.GET_ISSUE_DETAILS: lambda p: self._get_issue_details(p.issue_key, p),
                JiraActionType.LIST_EPICS: self._list_epics,
                JiraActionType.GET_SPRINT_INFO: self._get_sprint_info,
                JiraActionType.TICKETS_BY_ASSIGNEE: self._tickets_by_assignee,
                JiraActionType.OVERDUE_TICKETS: self._overdue_tickets,
                JiraActionType.TICKETS_BY_STATUS: self._tickets_by_status,
                JiraActionType.WORKLOG_SUMMARY: self._worklog_summary,
                JiraActionType.PRIORITY_BREAKDOWN: self._priority_breakdown,
                JiraActionType.BLOCKED_TICKETS: self._blocked_tickets,
            }

            if params.action not in action_map:
                raise ValueError(f"Unsupported action: {params.action}. Available actions: {list(action_map.keys())}")

            # Validate required parameters
            if params.action == JiraActionType.LATEST_TICKET_IN_EPIC and not params.epic_key:
                raise ValueError("epic_key is required for latest_ticket_in_epic")
            if params.action == JiraActionType.SEARCH_JQL and not params.jql:
                raise ValueError("jql is required for search_jql")
            if params.action == JiraActionType.GET_ISSUE_DETAILS and not params.issue_key:
                raise ValueError("issue_key is required for get_issue_details")

            logger.info(f"Executing action: {params.action}")
            resp = action_map[params.action](params)
            logger.info(f"Action completed successfully: {params.action}")
            yield ToolResponse(id=JIRA_STRUCTURED_RESPONSE_ID, response=resp)

        except Exception as e:
            logger.exception(f"Jira Structured Tool error: {e}")
            yield ToolResponse(
                id=JIRA_STRUCTURED_RESPONSE_ID,
                response={
                    "error": str(e),
                    "type": "error",
                    "action": llm_kwargs.get("action", "unknown"),
                    "parameters": llm_kwargs
                }
            )

    def _latest_ticket_in_epic(self, epic_key: str) -> Dict[str, Any]:
        """Get latest ticket in epic with enhanced information"""
        jql = f"(parent = {epic_key} OR 'Epic Link' = {epic_key}) ORDER BY updated DESC"
        issues = self.jira.search_issues(jql_str=jql, startAt=0, maxResults=1)

        if not issues:
            return {"type": "latest_ticket_in_epic", "jql": jql, "latest": None}

        issue = issues[0]
        return {
            "type": "latest_ticket_in_epic",
            "jql": jql,
            "latest": self._extract_issue_data(issue)
        }

    def _search_jql(self, jql: str, max_results: int) -> Dict[str, Any]:
        """Enhanced JQL search with better data extraction"""
        try:
            logger.info(f"Executing JQL query: {jql}")
            
            # Special handling for single issue key queries
            issue_key_match = re.match(r'^key\s*=\s*([A-Z]+-\d+)$', jql.strip(), re.IGNORECASE)
            if issue_key_match:
                issue_key = issue_key_match.group(1)
                logger.info(f"Detected single issue key query, using direct issue fetch for: {issue_key}")
                try:
                    issue = self.jira.issue(issue_key)
                    item = self._extract_issue_data(issue)
                    return {
                        "type": "search_jql",
                        "jql": jql,
                        "total_found": 1,
                        "returned": 1,
                        "items": [item]
                    }
                except JIRAError as direct_e:
                    logger.warning(f"Direct issue fetch failed for {issue_key}, falling back to JQL search: {direct_e}")
            
            issues = self.jira.search_issues(jql_str=jql, startAt=0, maxResults=max_results)
            items = [self._extract_issue_data(issue) for issue in issues]

            return {
                "type": "search_jql",
                "jql": jql,
                "total_found": issues.total,
                "returned": len(items),
                "items": items
            }
        except JIRAError as e:
            logger.error(f"JQL query failed: {jql}, Error: {e}")
            return {
                "type": "search_jql",
                "error": f"JQL query failed: {e}",
                "jql": jql,
                "details": str(e)
            }

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def description(self) -> str:
        return self._DESCRIPTION

    @property
    def display_name(self) -> str:
        return self._DISPLAY_NAME

    def build_tool_message_content(self, *args: ToolResponse) -> str | list[str | dict[str, Any]]:
        payload = args[-1].response if args else {}
        return json.dumps(payload, indent=2)

    def final_result(self, *args: ToolResponse):
        return args[-1].response if args else {}