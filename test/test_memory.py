"""
Exhaustive unit tests for the OpenManus-RL memory subsystem.

Covers:
- SimpleMemory  (openmanus_rl/memory/memory.py)
- SummarizedMemory (openmanus_rl/memory/summarized_memory.py)
- FileMemory    (openmanus_rl/memory/file_memory.py)
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from openmanus_rl.memory.memory import SimpleMemory
from openmanus_rl.memory.summarized_memory import SummarizedMemory, simple_summarize
from openmanus_rl.memory.file_memory import FileMemory


# ============================================================
# SimpleMemory
# ============================================================

class TestSimpleMemoryInit:
    def test_initial_state(self):
        mem = SimpleMemory()
        assert mem._data is None
        assert mem.keys is None
        assert mem.batch_size == 0

    def test_reset_sets_batch_size(self):
        mem = SimpleMemory()
        mem.reset(4)
        assert mem.batch_size == 4
        assert len(mem) == 4

    def test_reset_creates_empty_lists(self):
        mem = SimpleMemory()
        mem.reset(3)
        for i in range(3):
            assert mem[i] == []

    def test_reset_clears_existing_data(self):
        mem = SimpleMemory()
        mem.reset(2)
        mem.store({"obs": ["a", "b"], "act": ["x", "y"]})
        mem.reset(2)
        assert all(env == [] for env in mem._data)

    def test_reset_idempotent(self):
        mem = SimpleMemory()
        mem.reset(2)
        mem.reset(2)
        assert mem.batch_size == 2


class TestSimpleMemoryStore:
    def test_store_single_record(self):
        mem = SimpleMemory()
        mem.reset(2)
        mem.store({"text_obs": ["o1", "o2"], "action": ["a1", "a2"]})
        assert len(mem[0]) == 1
        assert mem[0][0]["text_obs"] == "o1"
        assert mem[1][0]["action"] == "a2"

    def test_store_multiple_records(self):
        mem = SimpleMemory()
        mem.reset(2)
        for i in range(5):
            mem.store({"text_obs": [f"o{i}_0", f"o{i}_1"], "action": [f"a{i}_0", f"a{i}_1"]})
        assert len(mem[0]) == 5
        assert len(mem[1]) == 5

    def test_store_preserves_key_order(self):
        mem = SimpleMemory()
        mem.reset(1)
        mem.store({"text_obs": ["obs"], "action": ["act"]})
        assert mem.keys == ["text_obs", "action"]

    def test_store_rejects_key_change(self):
        """Storing records with different keys should raise AssertionError."""
        mem = SimpleMemory()
        mem.reset(1)
        mem.store({"text_obs": ["obs"], "action": ["act"]})
        with pytest.raises(AssertionError):
            mem.store({"text_obs": ["obs"], "different_key": ["val"]})

    def test_store_single_env(self):
        mem = SimpleMemory()
        mem.reset(1)
        mem.store({"text_obs": ["hello"], "action": ["go north"]})
        assert mem[0][0]["text_obs"] == "hello"
        assert mem[0][0]["action"] == "go north"


class TestSimpleMemoryFetch:
    def _setup(self, batch_size: int, steps: int):
        mem = SimpleMemory()
        mem.reset(batch_size)
        for step in range(steps):
            obs = [f"obs{step}_{env}" for env in range(batch_size)]
            acts = [f"act{step}_{env}" for env in range(batch_size)]
            mem.store({"text_obs": obs, "action": acts})
        return mem

    def test_fetch_all_history(self):
        mem = self._setup(2, 3)
        contexts, lengths = mem.fetch(history_length=10)
        assert len(contexts) == 2
        assert all(l == 3 for l in lengths)

    def test_fetch_limited_history(self):
        mem = self._setup(2, 5)
        contexts, lengths = mem.fetch(history_length=2)
        assert all(l == 2 for l in lengths)

    def test_fetch_returns_formatted_strings(self):
        mem = self._setup(1, 2)
        contexts, _ = mem.fetch(history_length=10)
        assert "Observation" in contexts[0]
        assert "Action" in contexts[0]

    def test_fetch_empty_history(self):
        mem = SimpleMemory()
        mem.reset(2)
        contexts, lengths = mem.fetch(history_length=5)
        assert contexts == ["", ""]
        assert lengths == [0, 0]

    def test_fetch_custom_keys(self):
        mem = SimpleMemory()
        mem.reset(1)
        mem.store({"Observation": ["obs_val"], "Action": ["act_val"]})
        contexts, lengths = mem.fetch(history_length=5, obs_key="Observation", action_key="Action")
        assert lengths == [1]
        assert "obs_val" in contexts[0]

    def test_fetch_step_numbering_starts_at_one(self):
        mem = self._setup(1, 3)
        contexts, _ = mem.fetch(history_length=10)
        assert "Observation 1:" in contexts[0]

    def test_fetch_limited_window_correct_numbering(self):
        """Fetching a window from the middle should reflect correct step numbers."""
        mem = self._setup(1, 5)
        contexts, lengths = mem.fetch(history_length=2)
        assert lengths[0] == 2
        # Should start from step 4 since history_length=2 from 5 total
        assert "Observation 4:" in contexts[0]


class TestSimpleMemoryLenGetitem:
    def test_len_after_reset(self):
        mem = SimpleMemory()
        mem.reset(7)
        assert len(mem) == 7

    def test_getitem_returns_list(self):
        mem = SimpleMemory()
        mem.reset(3)
        assert isinstance(mem[2], list)

    def test_getitem_after_store(self):
        mem = SimpleMemory()
        mem.reset(2)
        mem.store({"k": ["v0", "v1"]})
        assert mem[0][0]["k"] == "v0"
        assert mem[1][0]["k"] == "v1"


# ============================================================
# SummarizedMemory
# ============================================================

class TestSummarizedMemoryInit:
    def test_inherits_from_simple_memory(self):
        assert issubclass(SummarizedMemory, SimpleMemory)

    def test_reset_initializes_summaries(self):
        mem = SummarizedMemory()
        mem.reset(3)
        assert mem.summaries == [None, None, None]
        assert mem.last_summary_step == [0, 0, 0]

    def test_reset_clears_old_summaries(self):
        mem = SummarizedMemory()
        mem.reset(2)
        mem.summaries = ["old", "old"]
        mem.reset(2)
        assert mem.summaries == [None, None]


class TestSummarizedMemoryFetchWithoutSummary:
    def test_fetch_delegates_to_simple_memory(self):
        mem = SummarizedMemory()
        mem.reset(2)
        mem.store({"text_obs": ["o1", "o2"], "action": ["a1", "a2"]})
        contexts, lengths = mem.fetch(history_length=10, use_summary=False)
        assert lengths == [1, 1]
        assert "o1" in contexts[0]

    def test_fetch_default_no_summary(self):
        """use_summary defaults to False – should behave identically to SimpleMemory."""
        mem = SummarizedMemory()
        mem.reset(1)
        mem.store({"text_obs": ["hello"], "action": ["world"]})
        contexts, lengths = mem.fetch(history_length=5)
        assert lengths == [1]


class TestSummarizedMemoryFetchWithSummary:
    def test_single_step_skips_summarization(self):
        """With only 1 step, no API call should be made."""
        mem = SummarizedMemory()
        mem.reset(1)
        mem.store({"text_obs": ["obs"], "action": ["act"]})

        with patch("openmanus_rl.memory.summarized_memory.simple_summarize") as mock_summ:
            contexts, lengths = mem.fetch(
                history_length=5,
                use_summary=True,
                summary_api_key=None,
            )
            mock_summ.assert_not_called()
        assert lengths[0] == 1

    def test_multi_step_calls_summarization(self):
        mem = SummarizedMemory()
        mem.reset(1)
        for i in range(3):
            mem.store({"text_obs": [f"obs{i}"], "action": [f"act{i}"]})

        with patch(
            "openmanus_rl.memory.summarized_memory.simple_summarize",
            return_value="SUMMARY",
        ) as mock_summ:
            contexts, lengths = mem.fetch(
                history_length=5,
                use_summary=True,
                summary_api_key="fake-key",
            )
            mock_summ.assert_called_once()
        assert contexts[0] == "SUMMARY"
        assert lengths[0] == 3

    def test_summary_cached_after_first_call(self):
        mem = SummarizedMemory()
        mem.reset(1)
        for i in range(2):
            mem.store({"text_obs": [f"obs{i}"], "action": [f"act{i}"]})

        with patch(
            "openmanus_rl.memory.summarized_memory.simple_summarize",
            return_value="CACHED_SUMMARY",
        ) as mock_summ:
            mem.fetch(history_length=5, use_summary=True, summary_api_key="k")
            # Second fetch with same steps should use cache
            mem.fetch(history_length=5, use_summary=True, summary_api_key="k")
            assert mock_summ.call_count == 1  # Called only once

    def test_summary_refreshed_after_new_store(self):
        mem = SummarizedMemory()
        mem.reset(1)
        for i in range(2):
            mem.store({"text_obs": [f"obs{i}"], "action": [f"act{i}"]})

        with patch(
            "openmanus_rl.memory.summarized_memory.simple_summarize",
            return_value="NEW_SUMMARY",
        ) as mock_summ:
            mem.fetch(history_length=5, use_summary=True, summary_api_key="k")
            # Add a new record then fetch again – cache should be invalidated
            mem.store({"text_obs": ["new_obs"], "action": ["new_act"]})
            mem.fetch(history_length=5, use_summary=True, summary_api_key="k")
            assert mock_summ.call_count == 2

    def test_extra_kwargs_ignored_gracefully(self):
        """SummarizedMemory.fetch should not raise on unknown kwargs."""
        mem = SummarizedMemory()
        mem.reset(1)
        mem.store({"text_obs": ["o"], "action": ["a"]})
        # Should not raise
        mem.fetch(history_length=5, use_summary=False, unknown_param="x")

    def test_concurrent_envs_summary(self):
        """Each environment in a batch gets its own summary."""
        mem = SummarizedMemory()
        mem.reset(2)
        for i in range(3):
            mem.store({"text_obs": [f"obs{i}_0", f"obs{i}_1"], "action": [f"act{i}_0", f"act{i}_1"]})

        call_count = [0]
        def fake_summ(history_steps, **kwargs):
            call_count[0] += 1
            return f"SUMMARY_{call_count[0]}"

        with patch("openmanus_rl.memory.summarized_memory.simple_summarize", side_effect=fake_summ):
            contexts, lengths = mem.fetch(
                history_length=5, use_summary=True, summary_api_key="k"
            )
        assert call_count[0] == 2  # One per env
        assert lengths == [3, 3]


class TestSimpleSummarize:
    def test_no_api_key_returns_fallback(self):
        steps = ["step1", "step2", "step3", "step4"]
        result = simple_summarize(steps, api_key=None)
        assert result  # non-empty
        # Fallback is last 3 steps
        assert "step4" in result

    def test_no_api_key_empty_history(self):
        result = simple_summarize([], api_key=None)
        assert result == ""

    def test_api_failure_returns_fallback(self):
        """When HTTP request fails, should fall back to last 3 steps."""
        steps = [f"step{i}" for i in range(6)]
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("network error")
            result = simple_summarize(steps, api_key="fake-key")
        assert "step5" in result  # last step should be there

    def test_api_error_status_returns_fallback(self):
        steps = ["a", "b", "c", "d"]
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("requests.post", return_value=mock_resp):
            result = simple_summarize(steps, api_key="fake-key")
        assert "d" in result  # last step in fallback

    def test_api_success_returns_content(self):
        steps = ["obs1", "obs2"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Summary text"}}]
        }
        with patch("requests.post", return_value=mock_resp):
            result = simple_summarize(steps, api_key="real-key")
        assert result == "Summary text"

    def test_webshop_env_type_prompt(self):
        """Webshop env type should not raise and use correct prompt path."""
        steps = ["step1"]
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("skip")
            # Should not raise even with webshop env_type
            result = simple_summarize(steps, api_key="k", env_type="webshop")
        assert result is not None

    def test_alfworld_env_type_prompt(self):
        steps = ["step1"]
        with patch("requests.post") as mock_post:
            mock_post.side_effect = Exception("skip")
            result = simple_summarize(steps, api_key="k", env_type="alfworld")
        assert result is not None

    def test_custom_model_passed_to_api(self):
        steps = ["step1"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            simple_summarize(steps, api_key="k", model="gpt-4-turbo")
            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "gpt-4-turbo"


# ============================================================
# FileMemory
# ============================================================

class TestFileMemoryInit:
    def test_inherits_simple_memory(self):
        assert issubclass(FileMemory, SimpleMemory)

    def test_default_memory_file(self):
        mem = FileMemory()
        assert mem.memory_file == "memory.md"

    def test_custom_memory_file(self):
        mem = FileMemory("custom.md")
        assert mem.memory_file == "custom.md"

    def test_init_missing_file_is_ok(self):
        """Initialising with a non-existent file should not raise."""
        mem = FileMemory("/tmp/nonexistent_42.md")
        assert mem.file_cache == []


class TestFileMemoryStoreToFile:
    def test_store_creates_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            mem = FileMemory(path)
            mem.store_to_file("hello", episode="ep1", step=0)
            with open(path) as fh:
                content = fh.read()
            assert "hello" in content
        finally:
            os.unlink(path)

    def test_store_updates_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            mem = FileMemory(path)
            mem.store_to_file("cached_entry", episode="e", step=1)
            assert any("cached_entry" in line for line in mem.file_cache)
        finally:
            os.unlink(path)

    def test_store_appends_to_file(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            mem = FileMemory(path)
            mem.store_to_file("first", step=0)
            mem.store_to_file("second", step=1)
            with open(path) as fh:
                content = fh.read()
            assert "first" in content
            assert "second" in content
        finally:
            os.unlink(path)

    def test_store_without_episode(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            mem = FileMemory(path)
            mem.store_to_file("no episode", step=3)
            with open(path) as fh:
                content = fh.read()
            assert "S:3" in content
        finally:
            os.unlink(path)


class TestFileMemoryQuery:
    def _make_mem(self):
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        return FileMemory(path), path

    def test_query_empty_memory(self):
        mem, path = self._make_mem()
        try:
            result = mem.query("anything")
            assert result == "No relevant memory found"
        finally:
            os.unlink(path)

    def test_query_finds_match_in_file(self):
        mem, path = self._make_mem()
        try:
            mem.store_to_file("pick up the red apple", step=1)
            result = mem.query("apple")
            assert "apple" in result
        finally:
            os.unlink(path)

    def test_query_no_match_returns_message(self):
        mem, path = self._make_mem()
        try:
            mem.store_to_file("irrelevant content", step=0)
            result = mem.query("dragon")
            assert result == "No relevant memory found"
        finally:
            os.unlink(path)

    def test_query_limit(self):
        mem, path = self._make_mem()
        try:
            for i in range(10):
                mem.store_to_file(f"apple item {i}", step=i)
            result = mem.query("apple", limit=3)
            # Should return at most 3 lines
            lines = [l for l in result.split("\n") if l.strip()]
            assert len(lines) <= 3
        finally:
            os.unlink(path)

    def test_query_searches_in_memory_data(self):
        mem, path = self._make_mem()
        try:
            mem.reset(1)
            mem.store({"text_obs": ["find the blue key"], "action": ["look"]})
            result = mem.query("blue key")
            assert "blue key" in result
        finally:
            os.unlink(path)


class TestFileMemoryClearAndRecent:
    def test_clear_file_empties_file(self):
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            mem = FileMemory(path)
            mem.store_to_file("data", step=0)
            mem.clear_file()
            with open(path) as fh:
                assert fh.read() == ""
            assert mem.file_cache == []
        finally:
            os.unlink(path)

    def test_get_recent_empty(self):
        mem = FileMemory("/tmp/no_such_file_xyz.md")
        assert mem.get_recent_from_file(5) == []

    def test_get_recent_returns_last_n(self):
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            mem = FileMemory(path)
            for i in range(10):
                mem.store_to_file(f"entry {i}", step=i)
            recent = mem.get_recent_from_file(3)
            assert len(recent) == 3
            assert "entry 9" in recent[-1]
        finally:
            os.unlink(path)


class TestFileMemoryStoreStaged:
    def test_store_staged_plan_to_file(self):
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            mem = FileMemory(path)
            mem.reset(1)
            mem.store_staged({"plan": "Go north", "action": "go north", "reflection": ""})
            with open(path) as fh:
                content = fh.read()
            assert "Go north" in content
        finally:
            os.unlink(path)

    def test_store_staged_reflection_to_file(self):
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            mem = FileMemory(path)
            mem.reset(1)
            mem.store_staged({"plan": "", "action": "look", "reflection": "Learned X"})
            with open(path) as fh:
                content = fh.read()
            assert "Learned X" in content
        finally:
            os.unlink(path)

    def test_store_staged_memory_store(self):
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            mem = FileMemory(path)
            mem.reset(1)
            mem.store_staged({"plan": "", "action": "", "memory_store": "Important fact"})
            with open(path) as fh:
                content = fh.read()
            assert "Important fact" in content
        finally:
            os.unlink(path)
