# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "llamabot[all]",
#   "pyprojroot",
#   "tqdm",
# ]
# ///

import time
from pathlib import Path

from llamabot.components.docstore import LanceDBDocStore
from pyprojroot import here
from tqdm import tqdm


def get_blog_lr_files(blog_path: Path) -> list[Path]:
    """Get all .lr files in the blog directory.

    Args:
        blog_path: Path to the blog directory

    Returns:
        List of Path objects pointing to .lr files
    """
    return list(blog_path.rglob("*.lr"))


docstore = LanceDBDocStore(
    table_name="ericmjl-blog-posts",
    storage_path=here() / "apis" / "embedder" / "lancedb",
)


# Get all blog posts
start_time = time.time()

blog_path = here() / "content" / "blog"
blog_posts = []

# Read all .lr files
for lr_file in tqdm(get_blog_lr_files(blog_path)):
    blog_posts.append(lr_file.read_text())


print(f"Found {len(blog_posts)} blog posts")
# Extend the docstore with the blog posts
docstore.extend(blog_posts)

end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")

# Performance notes:
# - First run: ~6s on MacBookPro M4Max with 128GB RAM
# - Subsequent runs: <2s due to LanceDB caching
# - Most time spent on initial embedding creation
# - LlamaBot's checks on whether a document has been embedded before
#   significantly speeds up subsequent runs
