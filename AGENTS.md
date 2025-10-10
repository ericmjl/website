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
my work make sure to keep the natural voice of the content

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
