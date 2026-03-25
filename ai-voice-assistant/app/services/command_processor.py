"""
NLP Command Processor using OpenAI API.

Parses transcribed speech into structured commands with:
- Intent classification across 12+ categories
- Entity extraction (dates, names, locations, etc.)
- Confidence scoring
- Context-aware multi-turn processing
- Natural language response generation for TTS
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandEntity:
    """An extracted entity from the command."""
    type: str       # entity type: date, time, location, person, query, etc.
    value: str      # extracted value
    confidence: float = 1.0


@dataclass
class ProcessedCommand:
    """Structured result of command processing."""
    intent: str
    entities: list[CommandEntity] = field(default_factory=list)
    response: str = ""
    confidence: float = 0.0
    raw_text: str = ""
    processing_time_ms: float = 0.0
    context_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "entities": [
                {"type": e.type, "value": e.value, "confidence": e.confidence}
                for e in self.entities
            ],
            "response": self.response,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "processing_time_ms": self.processing_time_ms,
            "context_used": self.context_used,
            "error": self.error,
        }


# Comprehensive intent definitions for the system prompt
INTENT_DEFINITIONS = """
Available intents and their descriptions:
- search: Web search or information lookup ("search for...", "what is...", "look up...")
- reminder: Set a reminder ("remind me to...", "don't forget to...")
- timer: Set a timer or alarm ("set a timer for...", "wake me up at...")
- weather: Weather queries ("what's the weather...", "will it rain...")
- email: Send or read emails ("send email to...", "check my email...")
- calendar: Calendar events ("schedule a meeting...", "what's on my calendar...")
- smart_home: Home automation ("turn on the lights", "set thermostat to...")
- music: Music playback ("play some jazz", "next song", "pause music")
- navigation: Directions and maps ("navigate to...", "how far is...")
- general_query: General knowledge questions not fitting other categories
- task_create: Create a task or to-do item ("add a task...", "create a to-do...")
- unknown: Cannot determine intent from the input
"""


class CommandProcessor:
    """
    Processes transcribed text into structured commands using OpenAI API.
    
    Supports:
    - Single-turn command processing
    - Multi-turn context-aware conversations
    - Batch command processing
    - Custom intent registration
    """

    def __init__(self, config):
        self.config = config
        self.api_key = config.api_key
        self.model = config.model
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.system_prompt = config.system_prompt + "\n" + INTENT_DEFINITIONS
        self.conversation_history: list[dict] = []
        self.max_history_turns = 5
        self._client = None

        logger.info("CommandProcessor initialised — model=%s", self.model)

    @property
    def client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key if self.api_key else None)
            except ImportError:
                logger.error("openai package not installed")
                raise
        return self._client

    def process(
        self,
        text: str,
        use_context: bool = False,
        additional_context: Optional[str] = None,
    ) -> ProcessedCommand:
        """
        Process transcribed text into a structured command.

        Args:
            text: The transcribed speech text.
            use_context: Whether to include conversation history.
            additional_context: Extra context (e.g., user preferences, location).

        Returns:
            ProcessedCommand with intent, entities, and response.
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return ProcessedCommand(
                intent="unknown",
                response="I didn't catch that. Could you please repeat?",
                confidence=0.0,
                raw_text=text or "",
                processing_time_ms=0.0,
            )

        try:
            messages = self._build_messages(text, use_context, additional_context)
            response = self._call_api(messages)
            command = self._parse_response(response, text)

            # Update conversation history
            if use_context:
                self._update_history(text, command.response)

            command.processing_time_ms = (time.perf_counter() - start_time) * 1000
            command.context_used = use_context

            logger.info(
                "Command processed: intent=%s, confidence=%.2f, time=%.0fms",
                command.intent, command.confidence, command.processing_time_ms,
            )
            return command

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.error("Command processing failed: %s", e)
            return ProcessedCommand(
                intent="unknown",
                response="I'm sorry, I had trouble processing that. Please try again.",
                confidence=0.0,
                raw_text=text,
                processing_time_ms=elapsed,
                error=str(e),
            )

    def process_batch(self, texts: list[str]) -> list[ProcessedCommand]:
        """Process multiple commands in sequence."""
        return [self.process(text) for text in texts]

    def _build_messages(
        self,
        text: str,
        use_context: bool,
        additional_context: Optional[str],
    ) -> list[dict]:
        """Build the messages array for the OpenAI API call."""
        system = self.system_prompt
        if additional_context:
            system += f"\n\nAdditional context: {additional_context}"

        messages = [{"role": "system", "content": system}]

        # Add conversation history for context
        if use_context and self.conversation_history:
            recent = self.conversation_history[-self.max_history_turns * 2:]
            messages.extend(recent)

        # Add current user message
        prompt = (
            f'Parse the following voice command and respond ONLY with valid JSON:\n'
            f'"{text}"\n\n'
            f'JSON format: {{"intent": "...", "entities": [{{"type": "...", "value": "...", '
            f'"confidence": 0.0-1.0}}], "response": "...", "confidence": 0.0-1.0}}'
        )
        messages.append({"role": "user", "content": prompt})

        return messages

    def _call_api(self, messages: list[dict]) -> str:
        """Make the OpenAI API call."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content

    def _parse_response(self, response: str, raw_text: str) -> ProcessedCommand:
        """Parse the API JSON response into a ProcessedCommand."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                return ProcessedCommand(
                    intent="unknown",
                    response="I understood your request but had trouble processing it.",
                    confidence=0.0,
                    raw_text=raw_text,
                )

        # Extract entities
        entities = []
        for entity_data in data.get("entities", []):
            entities.append(CommandEntity(
                type=entity_data.get("type", "unknown"),
                value=str(entity_data.get("value", "")),
                confidence=float(entity_data.get("confidence", 0.8)),
            ))

        return ProcessedCommand(
            intent=data.get("intent", "unknown"),
            entities=entities,
            response=data.get("response", ""),
            confidence=float(data.get("confidence", 0.0)),
            raw_text=raw_text,
        )

    def _update_history(self, user_text: str, assistant_response: str):
        """Update conversation history for multi-turn context."""
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        # Trim to max history
        max_entries = self.max_history_turns * 2
        if len(self.conversation_history) > max_entries:
            self.conversation_history = self.conversation_history[-max_entries:]

    def clear_context(self):
        """Reset conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation context cleared")

    def get_supported_intents(self) -> list[str]:
        """Return list of supported intent categories."""
        return [
            "search", "reminder", "timer", "weather", "email",
            "calendar", "smart_home", "music", "navigation",
            "general_query", "task_create", "unknown",
        ]
