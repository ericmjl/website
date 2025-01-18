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
