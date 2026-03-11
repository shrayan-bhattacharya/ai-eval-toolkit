"""
Project 2 - RAG QA Chatbot
Step 2: Text Chunking Module

Splits a long document into overlapping chunks for vector storage.
Chunk size: 500 characters | Overlap: 50 characters
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split text into overlapping chunks.

    Args:
        text:       The full document text to split.
        chunk_size: Target character length per chunk (default 500).
        overlap:    Characters shared between consecutive chunks (default 50).

    Returns:
        List of dicts: {"index": int, "text": str, "char_count": int}
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append({
            "index": index,
            "text": chunk,
            "char_count": len(chunk),
        })
        start += chunk_size - overlap
        index += 1

    return chunks


# ── Sample test ───────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Dollar General Corporation is one of America's largest and fastest-growing retailers,
operating more than 19,000 stores across 47 states as of 2024. Founded in 1939 by
J.L. Turner and his son Cal Turner Sr. in Scottsville, Kentucky, the company began as
a wholesale business before pivoting to retail. The "dollar general" concept — offering
everyday essential items at low, fixed price points — proved enormously popular with
budget-conscious shoppers in rural and suburban communities.

The company went public in 1968 and spent the following decades expanding aggressively
throughout the American South and Midwest. By the early 2000s, Dollar General had grown
into a billion-dollar enterprise, though rapid expansion also brought operational
challenges and a high-profile accounting restatement in 2007 that forced the company
to go private under the ownership of private equity firm KKR. The turnaround under
KKR's stewardship, led by CEO Rick Dreiling, focused on tightening store operations,
improving inventory management, and accelerating new store openings in underserved
markets. Dollar General returned to public markets in 2009 in one of the year's largest
retail IPOs.

Dollar General's business model targets what the company calls the "hardworking American"
— households earning less than $40,000 per year who rely on the stores for everyday
essentials including food, cleaning supplies, health and beauty products, and seasonal
merchandise. The stores are intentionally small (around 7,400 square feet on average)
and located in rural towns and small communities that are often underserved by Walmart,
Target, and other big-box competitors. This deliberate focus on rural America has been
a key competitive advantage and a major driver of the company's consistent growth.

In fiscal year 2023, Dollar General reported net sales of approximately $38.7 billion,
with more than 19,000 store locations and a workforce of approximately 163,000 employees.
The company's same-store sales growth and new store productivity have been closely
watched by Wall Street as indicators of the health of the lower-income consumer segment.
Despite macroeconomic headwinds including inflation and a pullback in government
stimulus, Dollar General has maintained its position as a go-to destination for
value-seeking shoppers.

Dollar General has also invested heavily in its supply chain and private-label brands.
The company's DG private label line spans hundreds of products across categories such
as food, household essentials, and personal care, often priced significantly below
national brand equivalents. In recent years, Dollar General has expanded its fresh and
refrigerated food offerings through its "DG Fresh" initiative, sourcing perishables
directly rather than through third-party distributors, which has improved both margins
and product availability. The company's distribution network includes more than 30
distribution centers strategically positioned to serve its vast store footprint.

Looking ahead, Dollar General's growth strategy centers on continued store expansion,
including its pOpshelf banner targeting higher-income suburban shoppers, and its
international push into Mexico through a partnership with local operators. The company
has also invested in technology upgrades including self-checkout kiosks, digital
couponing through its DG app, and data analytics capabilities to improve merchandising
decisions. While Dollar General faces increased scrutiny from regulators and labor
advocates regarding working conditions and wages, the company remains one of the most
closely watched retailers in the United States.
""".strip()


if __name__ == "__main__":
    print("=" * 60)
    print("TEXT CHUNKING MODULE — Project 2 RAG")
    print("=" * 60)
    print(f"Total input length : {len(SAMPLE_TEXT):,} characters")
    print(f"Chunk size         : 500 characters")
    print(f"Overlap            : 50 characters")
    print("=" * 60)

    chunks = chunk_text(SAMPLE_TEXT)

    print(f"\nProduced {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"--- Chunk {chunk['index']} ({chunk['char_count']} chars) ---")
        print(chunk["text"])
        print()

    print("=" * 60)
    print(f"Done. {len(chunks)} chunks ready for embedding.")
