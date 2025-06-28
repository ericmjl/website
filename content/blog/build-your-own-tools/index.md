## Why You Should Consider Building Your Own Tools

<blockquote>
  <p>Building your own tools is a liberating endeavor.<br>
  It injects joy back into your day-to-day work.<br>
  People were made to be creative creators.<br>
  Build your own tools</p>
</blockquote>


Slide 0

---

## A Flashback: Grad School Days

**Story Time**

![Circos Diagram](https://ericmjl.github.io/nxviz/images/circos.png)

*Does anyone know what this diagram is?*

I wanted to learn how to make a graph visualization like this. The only tool was in Perl, with no Python bindings—too complex for a beginner like me in 2014. So, I used what I knew: Python and matplotlib. This led to my first package, `circosplot`, in 2015. A year later, I could make all sorts of network visualizations!


Slide 1

---

## Empowerment Through Building

<div style="display: flex; flex-wrap: wrap; gap: 20px;">
  <img src="https://ericmjl.github.io/nxviz/examples/matrix/output_4_0.png" width="45%"/>
  <img src="https://ericmjl.github.io/nxviz/examples/geo/output_6_1.png" width="45%"/>
  <img src="https://ericmjl.github.io/nxviz/examples/arc_node_labels/output_2_1.png" width="45%"/>
  <img src="https://ericmjl.github.io/nxviz/examples/circos_node_labels/output_3_0.png" width="45%"/>
  <img src="https://ericmjl.github.io/nxviz/api/high-level-api/output_12_0.png" width="45%"/>
</div>

Being able to build my own Python package was superbly empowering, especially as a graduate student! I could build my own tools, archive them in the public domain, and never have to solve the same problem again. Echoes of Simon Willison:


Slide 2

---

## A Sneaky Way to Solve Problems Permanently

<blockquote>
  <p>I realized that one of the best things about open source software is that you can solve a problem once and then you can slap an open source license on that solution and you will <em>never</em> have to solve that problem ever again, no matter who’s employing you in the future.<br><br>
  It’s a sneaky way of solving a problem permanently.</p>
  <footer>— Simon Willison</footer>
</blockquote>

[Read more](https://simonwillison.net/2025/Jan/24/selfish-open-source/)


Slide 3

---

## Be the Change: pyjanitor

**2018: At Novartis**

Saw the R package `janitor` and thought: "Why can't Pythonistas have nice things?"

Remembered Gandhi's words:

> "Be the change you wish to see in the world"

So, I built **pyjanitor**. Now, Pythonistas can write expressive dataframe code:

```python
df = (
    pd.DataFrame.from_dict(company_sales)
    .remove_columns(["Company1"])
    .dropna(subset=["Company2", "Company3"])
    .rename_column("Company2", "Amazon")
    .rename_column("Company3", "Facebook")
    .add_column("Google", [450.0, 550.0, 800.0])
)
```

By being the change, we all benefit!


Slide 4

---

## Standardizing Tooling at Moderna

**2021: Joining Moderna**

- Pain points at Novartis: lack of standardization, hard onboarding
- At Moderna, we set standards: "Compute tasks" (dockerized CLI tools) and Python packages
- Designed project workflows around these deliverables
- Result: Easy collaboration, portable workflows, and rapid onboarding

> Eventually, tools that abstract away the Linux operating system will fail to satisfy users as they grow up and master Linux. They'll want to jump out of a container and just run raw Linux. Anything that tries to abstract away the filesystem, shell scripts, and more eventually runs into edge case, so why not just give people access to a raw linux machine with tools pre-installed? And when we build tools, why not just expose the abstractions in an open source manner?

— Paraphrased from Andrew Giessel


Slide 5

---

## Dogfooding and Internal Open Source

- Embraced a culture of building and improving our own tools
- Anyone can propose and contribute fixes
- We teach contributors how to do it "the right way"
- Result: Empowered, resilient teams not beholden to vendor roadmaps

<iframe width="560" height="315" src="https://www.youtube.com/embed/3ZTGwcHQfLY?si=_FLzvFyCp88ZlzGm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>


Slide 6

---

## Evolving the Tool Stack

We continually upgrade our tools:

- Switched to miniforge conda distribution
- Adopted `uv` and `pixi`
- Embraced Marimo notebooks
- Moved from `setup.py` to `pyproject.toml`
- Upgraded from `handlebars` to `cookiecutter`

**Key lesson:**

> There's no magic sauce that lies in the tool choices we make. The magic sauce is in the people who choose to show up and build. If your company has these types of people and empowers them to build things that are sensible to build (rather than buy).

<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7337223460651220992" height="349" width="504" frameborder="0" allowfullscreen title="Embedded post"></iframe>


Slide 7

---

## Building as a Path to Learning

![Graph Diagram](https://ericmjl.github.io/nxviz/examples/arc_node_labels/output_2_1.png)

**Building is a great way to learn new things.**

- Building nxviz taught me graph visualization principles
- Building LlamaBot taught me about LLM applications

> Computers are the best student there are. If you teach the computer something wrong, it'll give you back wrong answers. You just have to be good at verification, that's all.


Slide 8

---

## Iterating and Evolving: LlamaBot

- Created LlamaBot in 2023 to learn about LLMs and RAG
- Forced to encode understanding into code; rewrote it 4 times as my knowledge grew
- Key lessons:
  - The "Bot" analogy is a natural fit for agents
  - Abstractions should match the domain
  - Rewrites are normal and healthy

**Embrace the need to rewrite!**

With AI assistance, the barriers to building and iterating are lower than ever.


Slide 9

---

## Organizational Buy-In Matters

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are opening up a new role at Quora: a single engineer who will use AI to automate manual work across the company and increase employee productivity. I will work closely with this person. <a href="https://t.co/iKurWS6W7v">pic.twitter.com/iKurWS6W7v</a></p>&mdash; Adam D&#39;Angelo (@adamdangelo) <a href="https://twitter.com/adamdangelo/status/1936504553916309617?ref_src=twsrc%5Etfw">June 21, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

![Quora AI Automation](https://pbs.twimg.com/media/Gt_VT5nakAANTdj?format=png&name=large)

**Internal tool building requires organizational buy-in.**

- Does your organization empower you to build?
- Leadership support is crucial for innovation and productivity.


Slide 10

---

## Expert Advice: Build Custom Tools

<blockquote>
  <p><strong>Build a custom annotation tool.</strong> This is the single most impactful investment you can make for your AI evaluation workflow. With AI-assisted development tools like Cursor or Lovable, you can build a tailored interface in hours. I often find that teams with custom annotation tools iterate ~10x faster.</p>
  <ul>
    <li>They show all your context from multiple systems in one place</li>
    <li>They can render your data in a product specific way (images, widgets, markdown, buttons, etc.)</li>
    <li>They’re designed for your specific workflow (custom filters, sorting, progress bars, etc.)</li>
  </ul>
  <p>Off-the-shelf tools may be justified when you need to coordinate dozens of distributed annotators with enterprise access controls. Even then, many teams find the configuration overhead and limitations aren’t worth it.</p>
  <p>[Isaac’s Anki flashcard annotation app](https://youtu.be/fA4pe9bE0LY) shows the power of custom tools—handling 400+ results per query with keyboard navigation and domain-specific evaluation criteria that would be nearly impossible to configure in a generic tool.</p>
  <p>With AI-assisted development tools like Cursor or Lovable, you can build a tailored interface in hours.</p>
  <footer>— Hamel Husain</footer>
</blockquote>

**The barrier to entry for building your own tools is lower than ever!**


Slide 11

---

## Scaling Through Tooling

**Lessons from 10 Years of Building Tools:**

1. Software scales our labour.
2. Documentation scales our brains.
3. Tests scale others' trust in our code.
4. Design scales our agility.
5. Agents scale our processes.
6. Open source scales opportunity for impact.

If you build tools for yourself, you scale yourself. If you teach others, you scale their labour. Document well, and you scale your brain. Test thoroughly, and you scale trust. Design well, and you scale agility. Use agents, and you scale processes. Open source, and you scale opportunity.

> There are people dying, if you care enough for the living, make a better place for you and for me.
>
> -- Heal the World (Michael Jackson)


Slide 12
