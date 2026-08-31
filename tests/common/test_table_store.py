import json

import pytest

from common.table_store import load_tables_by_key


def test_streaming_table_store_loads_requested_keys_and_rejects_missing(tmp_path):
    path = tmp_path / "tables.jsonl"
    path.write_text("\n".join(json.dumps({"table_key": x}) for x in ("a|1", "b|2")), encoding="utf-8")
    assert set(load_tables_by_key(path, {"b|2"})) == {"b|2"}
    with pytest.raises(KeyError, match="missing"):
        load_tables_by_key(path, {"c|3"})
