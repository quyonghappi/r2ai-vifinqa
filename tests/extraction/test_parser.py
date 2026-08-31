from pathlib import Path

from extraction.parser import (
    derive_report_id,
    expand_grid,
    extract_corpus,
    extract_tables_from_file,
    TableGridParser,
    diagnose_candidate_shape,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "SYN_financial_statements_2020_consolidated_extracted.txt"


def test_derive_report_id_uses_parent_directory_name():
    # matches the real corpus layout: .../<DOC_NAME>/<DOC_NAME>_extracted.txt
    path = "financial_statements/AAA/2015/AAA_financial_statements_2015_consolidated/AAA_financial_statements_2015_consolidated_extracted.txt"
    assert derive_report_id(path) == "AAA_financial_statements_2015_consolidated"


def test_extract_corpus_report_ids_override_avoids_flattened_path_bug():
    # a flattened local cache (no parent-dir structure preserved) must not silently produce
    # a wrong report_id -- callers should pass the original corpus-relative path instead.
    flattened_local_path = str(FIXTURE)  # lives directly under tests/fixtures/, not a DOC_NAME dir
    original_relative_path = "financial_statements/SYN/2020/SYN_financial_statements_2020_consolidated/SYN_financial_statements_2020_consolidated_extracted.txt"
    correct_report_id = derive_report_id(original_relative_path)

    candidates = extract_corpus([flattened_local_path], report_ids=[correct_report_id])
    assert all(c.report_id == "SYN_financial_statements_2020_consolidated" for c in candidates)


def test_provenance_source_path_is_threaded_without_changing_read_path():
    source = "financial_statements/SYN/2020/doc/doc_extracted.txt"
    candidates = extract_corpus(
        [str(FIXTURE)],
        report_ids=["SYN_financial_statements_2020_consolidated"],
        provenance_source_paths=[source],
    )
    assert all(c.source_path == source for c in candidates)


def test_extracts_four_table_candidates_with_line_positions():
    candidates = extract_tables_from_file(str(FIXTURE))
    assert len(candidates) == 4
    # line_position must point at the actual line containing the <table> tag
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    for c in candidates:
        assert "<table>" in lines[c.line_position - 1]


def test_section_header_classification_uses_running_state():
    candidates = extract_tables_from_file(str(FIXTURE))
    assert candidates[0].section_header == "balance_sheet"
    assert candidates[1].section_header == "balance_sheet"  # "TÌNH HÌNH TÀI CHÍNH" also matches
    assert candidates[2].section_header == "notes"
    assert candidates[3].section_header == "notes"


def test_rowspan_colspan_grid_is_rectangular():
    candidates = extract_tables_from_file(str(FIXTURE))
    bank_table = candidates[1]
    assert bank_table.status == "success"
    row_lengths = {len(r) for r in bank_table.grid}
    assert row_lengths == {5}
    assert bank_table.grid[0] == ["", "", "Thuyết minh", "Tại ngày", "Tại ngày"]
    assert bank_table.grid[2] == ["A", "TÀI SẢN", "", "", ""]


def test_malformed_table_is_recovered_not_silently_wrong():
    candidates = extract_tables_from_file(str(FIXTURE))
    malformed = candidates[3]
    # The dangling <td> (missing </td> before the next <td>) is recovered by auto-closing
    # it -- but this must be flagged, not treated identically to a clean parse, and the
    # recovered text must not be silently dropped.
    assert malformed.status == "parsed_with_warnings"
    assert any("unclosed" in w for w in malformed.warnings)
    assert malformed.n_cols == 2
    assert malformed.grid[0] == ["Cột A", "Cột B"]


def test_expand_grid_handles_empty_input():
    assert expand_grid([]) == []


def test_table_grid_parser_does_not_raise_on_unknown_tag():
    p = TableGridParser()
    p.feed("<table><tr><td>a<fcel/>b</td></tr></table>")
    p.close()
    assert p.rows[0][0]["text"] == "ab"


def test_shape_anomalies_are_observable_but_do_not_drop_candidate():
    candidate = extract_tables_from_file(str(FIXTURE))[0]
    candidate.n_rows = 201
    candidate.n_cols = 51
    assert diagnose_candidate_shape(candidate) == ["n_rows>200:201", "n_cols>50:51"]
    assert candidate.status == "success"
