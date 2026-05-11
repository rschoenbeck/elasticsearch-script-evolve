"""make_client returns a configured Elasticsearch client; no module-level globals."""

from elasticsearch import Elasticsearch

from es_script_agent import es_client


def test_make_client_default_url() -> None:
    client = es_client.make_client()
    assert isinstance(client, Elasticsearch)


def test_make_client_explicit_url() -> None:
    client = es_client.make_client("http://example.invalid:9200")
    assert isinstance(client, Elasticsearch)


def test_no_module_level_client() -> None:
    # The module exposes a factory, not a pre-built client.
    public_attrs = {n for n in dir(es_client) if not n.startswith("_")}
    assert "make_client" in public_attrs
    for name in public_attrs:
        value = getattr(es_client, name)
        assert not isinstance(value, Elasticsearch), (
            f"es_client.{name} is a module-level Elasticsearch instance; "
            "callers should build their own via make_client()."
        )
