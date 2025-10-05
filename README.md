# Eric's Personal Website Repository

This is the personal website of Eric J. Ma. It is built in Lektor.

All content here, unless otherwise stated, belongs under the copyright of Eric J. Ma. Permission is required to host elsewhere.

## Adding Diagrams with Mermaid.js

This site supports Mermaid.js diagrams. To add a diagram in your blog posts or pages, use the following HTML syntax:

```html
<pre class="mermaid">
graph TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Do Something]
    B -->|No| D[Do Nothing]
</pre>
```

Note: Use the HTML syntax above rather than Markdown code fences, as Mermaid needs the specific `class="mermaid"` attribute to work properly.

For more information on Mermaid.js diagram syntax, visit the [Mermaid.js documentation](https://mermaid.js.org/syntax/flowchart.html).

## Commenting System

This website uses [giscus](https://giscus.app/) for comments, powered by GitHub Discussions.

## Templates

Templates are found in the `templates/` directory and define the structure and layout of different page types:

### Main Templates
- `layout.html` - Base template with HTML structure, navigation, and dark mode toggle
- `page.html` - Basic page template
- `blog.html` - Blog listing page
- `blog-post.html` - Individual blog post template
- `project.html` - Individual project page
- `projects.html` - Projects listing page
- `resume.html` - Resume/CV page
- `tag.html` - Tag page for blog post filtering
- `all-tags.html` - All tags listing page

### Template Components
- `macros/` - Reusable template macros for common functionality:
  - `blog.html` - Blog-related macros
  - `pagination.html` - Pagination controls
  - `project.html` - Project display macros
  - `slideshow.html` - Image slideshow functionality
- `blocks/` - Content blocks for the resume page:
  - `book.html` - Book information display
  - `description.html` - Description blocks
  - `github.html` - GitHub repository information
  - `notebooks.html` - Jupyter notebook links
  - `resource.html` - Resource links
  - `resume-education.html` - Education section
  - `resume-experience.html` - Work experience section
  - `resume-skills.html` - Skills section
  - `slides.html` - Presentation slides
  - `youtube.html` - YouTube video embeds
