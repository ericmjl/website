# AGENTS.md

For ad-hoc scripts, in-line script metadata can be used to provide Python
versions. In-line Script metadata looks like this:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///
```

I usually like to be at least 1 minor version behind the latest, and the
latest Python version as of 2024-12-19 is Python 3.13. For any scripts that
I ask you to make, make sure you also include the necessary dependencies.

I usually like to dictate my blog post content in my most natural voice. That
usually ends up being a little bit informal. If I ask you to edit in polish
my work make sure to keep the natural voice of the content. I do not use em
dashes (—); use commas, periods, or separate sentences instead. Study my
recent published blog posts for sentence rhythm and flow when editing.

## Terminal.css Dark Mode Implementation Notes

## 1. No Built-in Toggle
Terminal.css does **not** provide a built-in dark mode toggle via class or attribute. You must override CSS variables yourself for dark mode.

## 2. How to Implement Dark Mode
- Add a `.dark-mode` class to `<body>` when toggling dark mode.
- In your custom CSS, override Terminal.css variables inside `.dark-mode`, for example:

```css
body.dark-mode {
  --background-color: #222225;
  --font-color: #e8e9ed;
  --primary-color: #62c4ff;
  /* ...etc (see https://terminalcss.xyz/dark/) */
}
```

## 3. Custom UI Elements (e.g., Pagination Buttons)
- For custom UI elements that use non-terminal.css classes, add dark mode overrides in your custom CSS, using the same variables.
- Example for pagination:

```css
body.dark-mode .btn-primary.disabled {
  color: #fff !important;
}
body.dark-mode .btn, ... { ... }
```

## 4. Toggle Logic
- The toggle logic should add/remove `.dark-mode` on `<body>` and persist the preference in `localStorage`.

## 5. Maintainability
- **Do NOT** put dark mode CSS directly in `layout.html`; keep it in `custom.css` for maintainability.

## Language to Avoid

When generating text, avoid the following categories of wording, structures, and symbols:

## 1. Grandiose or clichéd phrasing

- "stands as", "serves as", "is a testament"
- "plays a vital / significant / crucial role"
- "underscores its importance", "highlights its significance"
- "leaves a lasting impact", "watershed moment", "deeply rooted",
  "profound heritage"
- "indelible mark", "solidifies", "rich cultural heritage / tapestry",
  "breathtaking"
- "must-visit / must see", "stunning natural beauty", "enduring / lasting
  legacy", "nestled", "in the heart of"

## 2. Formulaic rhetorical scaffolding

- "it's important to note / remember / consider"
- "it is worth …"
- "no discussion would be complete without …"
- "In summary", "In conclusion", "Overall"
- "Despite its … faces several challenges …"
- "Future Outlook", "Challenges and Legacy"
- "Not only … but …", "It is not just about … it's …"
- Rule-of-three clichés like "the good, the bad, and the ugly"

## 3. Empty attributions and hedges

- "Industry reports", "Observers have cited", "Some critics argue"
- Vague sources: "some argue", "some say", "some believe"
- "as of [date]", "Up to my last training update"
- "While specific details are limited / scarce", "not widely available /
  documented / disclosed", "based on available information"

## 4. AI disclaimers and meta-references

- "As an AI language model …", "as a large language model …"
- "I'm sorry …"
- "I hope this helps", "Would you like …?", "Let me know"
- Placeholder text such as "[Entertainer's Name]"

## 5. Letter-like or conversational boilerplate

- "Subject: …", "Dear …"
- "Thank you for your time / consideration"
- "I hope this message finds you well"
- "I am writing to …"

## 6. Stylistic markers of AI text

- Overuse of boldface for emphasis
- Bullets with bold headers followed by colons
- Emojis in headings or lists
- Overuse of em dashes (—) in place of commas/colons
- Inconsistent curly vs. straight quotation marks
- "From … to …" constructions when not a real range
- Unnecessary Markdown or formatting in plain-text contexts

## YouTube Content Management

### Using yt-dlp for Content Extraction
- **Always use `uvx yt-dlp`** to extract video metadata and content
- Command format: `uvx yt-dlp -j "https://www.youtube.com/watch?v=VIDEO_ID"`
- This provides accurate titles, descriptions, durations, and other metadata
- Much more reliable than web scraping or manual entry

### Content Organization Structure
- **Talks** (shorter presentations, < 1 hour): Store in `content/talks/`
- **Tutorials** (longer educational sessions, 2+ hours): Store in `content/teaching/`
- Both use the `projects` model but are organized by directory structure
- Always check for duplicates by searching existing YouTube video IDs before adding new content

### Duplicate Prevention
- Search for existing video IDs using: `grep -r "youtube\.com/watch\?v=" content/`
- Check both video ID and full URL patterns
- Never add duplicate content - always verify uniqueness first

### Content Entry Format
- Use accurate titles and descriptions from yt-dlp output
- Include proper sort_key for ordering
- Set appropriate category (usually "Work" for conference content)
- Include visible: Visible for public content
- Use descriptive directory names that match the content

### Title Format for Talks and Teaching
- **Standard format**: `<content> | <venue>` (e.g., "Bayesian Data Science by Simulation | SciPy 2020")
- **Do NOT include "Tutorial" or "Talk" in the title** - the venue and content should be clear enough
- Remove author names from titles (e.g., "Network Analysis Made Simple | SciPy 2022" not "Network Analysis Made Simple - Eric Ma, Mridul Seth | SciPy 2022")
- If venue is missing, extract from YouTube video metadata (channel name, description, or upload date context)
- Apply this format consistently to both `content/talks/` and `content/teaching/` entries

## Lektor .lr File Format Requirements

### Field Separators
- **CRITICAL**: Each field in .lr files must be separated by `---` on its own line
- Format: `field_name: value` followed by `---` before the next field
- When programmatically adding fields, always ensure proper separator formatting
- Missing separators will cause Lektor parsing errors

### Date Fields for Ordering
- Use `pub_date` field with `type = date` in model definitions
- Order content using `order_by = -pub_date, title` in parent model
- Extract dates from YouTube using `uvx yt-dlp -j <url>` and parse `upload_date` field
- Convert YYYYMMDD format to YYYY-MM-DD for Lektor compatibility

### Internal Links to Blog Posts
- Blog post URLs are **date-based**: `/blog/YYYY/M/d/post-slug/` (no leading zeros on month or day)
- The blog model uses `slug_format = (pub_date|dateformat('y/M/d/')) ~ this._id` (see `models/blog.ini`)
- When linking to another blog post from within a post, use the **relative** path with that post's `pub_date` and folder name (the `_id`): `/blog/YYYY/M/d/post-slug/`
- Example: for a post in `content/blog/exploratory-data-analysis-isnt-open-ended/` with `pub_date: 2024-01-28`, the link is `/blog/2024/1/28/exploratory-data-analysis-isnt-open-ended/`
- To get the correct URL for a post: read its `contents.lr` for `pub_date`, use the folder name as the slug; format as YYYY/M/d (e.g. 2024-01-28 → 2024/1/28)

## Development Workflow

### Running Python Commands
- **Always use `pixi run`** to execute Python commands to ensure the correct environment from `pyproject.toml` is used
- Example: `pixi run python script.py` instead of `python script.py`
- This ensures all dependencies defined in pixi features (e.g., blogbot, lektor) are available
- The project uses pixi for environment management with feature-based dependencies defined in `pyproject.toml`
