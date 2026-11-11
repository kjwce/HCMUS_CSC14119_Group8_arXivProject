"""Explicit flat Arrow schemas from the official M1, M2 and M3 statements."""

import pyarrow as pa


TEXT = pa.string()
TEXT_LIST = pa.list_(pa.field("item", TEXT, nullable=False))
CATEGORY = pa.dictionary(pa.int32(), TEXT)


SCHEMAS = {
    "papers": pa.schema([
        pa.field("arxiv_id", TEXT, nullable=False),
        pa.field("title", TEXT),
        pa.field("authors", TEXT_LIST),
        pa.field("submission_date", TEXT),
        pa.field("revised_dates", TEXT_LIST),
        pa.field("venue", TEXT),
    ]),
    "references": pa.schema([
        pa.field("arxiv_id", TEXT, nullable=False),
        pa.field("ref_arxiv_id", TEXT, nullable=False),
        pa.field("ref_title", TEXT),
        pa.field("ref_authors", TEXT_LIST),
        pa.field("ref_submission_date", TEXT),
        pa.field("ref_s2_id", TEXT),
    ]),
    "elements": pa.schema([
        pa.field("element_id", TEXT, nullable=False),
        pa.field("arxiv_id", TEXT, nullable=False),
        pa.field("kind", CATEGORY),
        pa.field("content", TEXT),
    ]),
    "hierarchy": pa.schema([
        pa.field("arxiv_id", TEXT, nullable=False),
        pa.field("version", pa.int32()),
        pa.field("element_id", TEXT, nullable=False),
        pa.field("parent_id", TEXT, nullable=False),
    ]),
    "bibtex": pa.schema([
        pa.field("arxiv_id", TEXT, nullable=False),
        pa.field("citation_key", TEXT, nullable=False),
        pa.field("entry_type", CATEGORY),
        pa.field("title", TEXT),
        pa.field("authors", TEXT_LIST),
        pa.field("year", pa.int32()),
        pa.field("doi", TEXT),
        pa.field("journal", TEXT),
        pa.field("url", TEXT),
    ]),
    "predictions": pa.schema([
        pa.field("arxiv_id", TEXT, nullable=False),
        pa.field("partition", CATEGORY),
        pa.field("citation_key", TEXT, nullable=False),
        pa.field("groundtruth_arxiv_id", TEXT),
        *[pa.field(f"cand_{i}", TEXT) for i in range(1, 6)],
    ]),
}

BASE_TABLES = tuple(name for name in SCHEMAS if name != "predictions")


def describe_schemas():
    return {
        name: [{"name": field.name, "type": str(field.type), "nullable": field.nullable}
               for field in schema]
        for name, schema in SCHEMAS.items()
    }
