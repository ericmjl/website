For ad-hoc scripts, in-line script metadata can be used to provide Python versions.
In-line Script metadata looks like this:

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

I usually like to be at least 1 minor version behind the latest,
and the latest Python version as of 2024-12-19 is Python 3.13.
For any scripts that I ask you to make, make sure you also include the necessary dependencies.

I usually like to dictate my blog post content in my most natural voice.
That usually ends up being a little bit informal.
If I ask you to edit in polish my work make sure to keep the natural voice of the content

# Terminal.css Dark Mode Implementation Notes

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
