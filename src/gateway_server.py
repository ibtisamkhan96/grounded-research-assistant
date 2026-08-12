"""The AI Gateway, as an MCP server.

Trusted literature exposed over the Model Context Protocol, the same pattern Wiley
uses with Anthropic to bring research into Claude. It reuses the trusted-content
layer, so the gateway and the agent always read the identical sources.

Run it standalone:
    python -m src.gateway_server

Register it in an MCP client (e.g. Claude Desktop) with that command, and any
client can search trusted literature through one endpoint.
"""
from mcp.server.fastmcp import FastMCP

from . import content

mcp = FastMCP("Research-Gateway")


@mcp.tool()
def search_literature(query: str, source: str = "openalex", k: int = 5) -> list:
    """Search trusted scholarly literature.

    source = "openalex" (peer-reviewed) or "arxiv" (preprints). Returns papers with title,
    authors, year, venue, DOI, and abstract, the trusted content an AI answer can be grounded
    in and cited to.
    """
    return content.search_arxiv(query, k) if source == "arxiv" else content.search_openalex(query, k)


if __name__ == "__main__":
    mcp.run()
