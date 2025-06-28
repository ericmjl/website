# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anthropic==0.54.0",
#     "ipython==9.3.0",
#     "llamabot[all]==0.12.10",
#     "marimo",
#     "openai==1.86.0",
#     "pydantic==2.11.5",
# ]
# ///

import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Build your own tools!

    **Alternatively titled:** I hate making slides, so let me try to conquer Murphy's law in front of you 🙃
    """  # noqa: E501
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""<iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7335296923001643008?collapsed=0" height="265" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>"""  # noqa: E501
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Making slides manually feels especially painful now that you know Cursor for slides should exist but doesn’t.</p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/1931042840966222046?ref_src=twsrc%5Etfw">June 6, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>"""  # noqa: E501
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## NERDSNIPE!

    So I went ahead and built it.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import os
    import re
    import tempfile
    from pathlib import Path
    from typing import Literal

    import llamabot as lmb
    import marimo as mo
    import openai
    from pydantic import BaseModel, Field, model_validator

    return (
        BaseModel,
        Field,
        Literal,
        Path,
        lmb,
        mo,
        model_validator,
        openai,
        os,
        re,
        tempfile,
    )


@app.cell(hide_code=True)
def _(BaseModel, Field, Literal, model_validator, re):
    class Slide(BaseModel):
        title: str
        content: str = Field(
            description="Arbitrary markdown or HTML content. Headers are not allowed in slide content. Use regular text formatting instead."  # noqa: E501
        )
        type: Literal["HTML", "Markdown"]  # noqa: F821

        @model_validator(mode="after")
        def check_no_header_in_content(self):
            """Check that there are no headers in the content, this is explicitly not allowed."""  # noqa: E501
            # Check for Markdown headers (# Header, ## Header, etc.)
            if self.type == "Markdown":
                # Look for lines starting with one or more # followed by a space
                header_pattern = re.compile(r"^#{1,6}\s", re.MULTILINE)
                if header_pattern.search(self.content):
                    raise ValueError(
                        "Headers are not allowed in slide content. Use regular text formatting instead."  # noqa: E501
                    )

            # For HTML content, check for header tags
            elif self.type == "HTML":
                header_tags = ["<h1", "<h2", "<h3", "<h4", "<h5", "<h6"]
                for tag in header_tags:
                    if tag in self.content.lower():
                        raise ValueError(
                            "HTML header tags (h1-h6) are not allowed in slide content."  # noqa: E501
                        )

            return self

        def render(self):
            return f"""## {self.title}

    {self.content}
            """

    return (Slide,)


@app.cell
def _(Slide, lmb):
    @lmb.prompt("system")
    def slidemaker_sysprompt():
        """You are an expert at making markdown slides.

        Your job is to produce a single slide that represents the content
        that a user asks for.
        Tables should be in HTML.
        """

    slidemaker = lmb.StructuredBot(slidemaker_sysprompt(), pydantic_model=Slide)
    return (slidemaker,)


@app.cell
def _(mo, slidemaker):
    eatwell_slide = slidemaker("Why eating well is so important")
    mo.md(eatwell_slide.render())
    return


@app.cell
def _():
    # sales_slide = slidemaker("Table showing growth in sales.")
    # mo.md(sales_slide.render())
    return


@app.cell
def _(mo, slidemaker):
    two_column_slide = slidemaker("Pros and cons of buying a thing")
    mo.md(two_column_slide.render())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Ok, let's now try to make an entire slide deck for today's talk.""")
    return


@app.cell(hide_code=True)
def _(BaseModel, Path, Slide, lmb, slidemaker):
    @lmb.prompt("user")
    def slidemaker_edit(new_request, existing_slide):
        """This is the request for an edit on a slide.

        {{ new_request }}

        The existing content is here:

        {{ existing_slide }}

        Help me create a new slide based on the new request,
        using the existing slide as inspiration or basis where appropriate.
        """

    @lmb.prompt("user")
    def slidemaker_insert(
        new_request: str,
        existing_slides: str,
    ):
        """This is the request to insert a slide.

        {{ new_request }}

        ---

        Here is the current state of the slides:

        {{ existing_slides }}

        ---

        Help me create a new slide based on the new request,
        weaving it seamlessly with the slide before and after.
        """

    class SlideDeck(BaseModel):
        slides: list[Slide]
        talk_title: str

        def render(self) -> str:
            """Render all slides as markdown with slide separators."""
            markdown_content = []

            for i, slide in enumerate(self.slides):
                # Add slide content
                slide_content = f"## {slide.title}\n\n{slide.content}"
                markdown_content.append(slide_content)
                markdown_content.append(f"\nSlide {i}")

                # Add separator after each slide except the last one
                if i < len(self.slides) - 1:
                    markdown_content.append("---")

            # Join all slides with newlines
            return "\n\n".join(markdown_content)

        def save(self, path: Path):
            """Save the slide deck as a markdown file."""
            markdown_content = self.render()

            # Ensure the directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the content to the file
            with open(path, "w", encoding="utf-8") as f:
                f.write(markdown_content)

            return f"Slide deck saved to {path}"

        def edit(self, index: int, change: str):
            """Edit the slide at a given index."""
            current_slide = self.slides[index].render()

            new_slide = slidemaker(slidemaker_edit(change, current_slide))
            self.slides[index] = new_slide

        def select(self, description: str):
            """Return a slide's index by natural language."""
            docstore = lmb.LanceDBDocStore(
                table_name="deckbot", storage_path=Path("/tmp")
            )
            docstore.reset()
            docstore.extend([slide.render() for slide in self.slides])
            index = {slide.render(): i for i, slide in enumerate(self.slides)}

            docs = docstore.retrieve(description)
            return index[docs[0]]

        def insert(self, index, description):
            """Insert a slide just before a given index."""
            current_slides = self.render()  # used for context
            new_slide = slidemaker(slidemaker_insert(description, current_slides))
            self.slides.insert(index, new_slide)

    return (SlideDeck,)


@app.cell
def _(SlideDeck, lmb):
    @lmb.prompt("system")
    def deckbot_sysprompt():
        """You are a bot that helps people make slides for a presentation.
        Vary the style of slides when you generate them, do not stick to only bullet points.
        Quotes should be formatted as such.
        If there are quotes that are provided to you in the request,
        ensure that they are placed in the slides.
        """  # noqa: E501

    chat_memory = lmb.LanceDBDocStore(table_name="deckbot-chat-memory")
    chat_memory.reset()

    deckbot = lmb.StructuredBot(
        system_prompt=deckbot_sysprompt(),
        pydantic_model=SlideDeck,
        chat_memory=chat_memory,
    )
    return (deckbot,)


@app.cell(hide_code=True)
def _():
    talk_source = """
    # Why you should consider building your own tools

    ---

    > Building your own tools is a liberating endeavor.
    > It injects joy back into your day-to-day work.
    > People were made to be creative creators.
    > Build your own tools

    ---

    Flashback: a story from my grad school days.

    ![](https://ericmjl.github.io/nxviz/images/circos.png)

    Anybody know what this diagram is?

    I wanted to learn how to make a graph visualization like this. But the only tool I saw out there was written in a different language (Perl), had no Python bindings, and was way too complicated for me, a beginner programmer in 2014, to learn. So I decided to use what I knew at the time -- Python and matplotlib -- to make my own Python package. Both to learn how to do software development, and to learn the principles of rational network visualization.

    The precursor to `nxviz`, `circosplot`, was born in 2015, and one year later, I knew enough that I could make all sorts of network visualizations!

    LIke this:

    ![](https://ericmjl.github.io/nxviz/examples/matrix/output_4_0.png)

    Or this:

    ![](https://ericmjl.github.io/nxviz/examples/geo/output_6_1.png)

    Or this:

    ![](https://ericmjl.github.io/nxviz/examples/arc_node_labels/output_2_1.png)

    Or this:

    ![](https://ericmjl.github.io/nxviz/examples/circos_node_labels/output_3_0.png)

    Or this:

    ![](https://ericmjl.github.io/nxviz/api/high-level-api/output_12_0.png)

    Being able to build my own Python package was superbly empowering, especially as a graduate student! I could build my own tools, archive them in the public domain, and never have to solve the same problem again. Echoes of Simon Willison.

    https://simonwillison.net/2025/Jan/24/selfish-open-source/

    > I realized that one of the best things about open source software is that you can solve a problem once and then you can slap an open source license on that solution and you will _never_ have to solve that problem ever again, no matter who’s employing you in the future.
    >
    > It’s a sneaky way of solving a problem permanently.

    ---

    Fast-forward to 2018, when I was working at Novartis. My colleague Brant Peterson showed me the R package `janitor`, and I was like, "why can't Pythonistas have nice things?"

    I then remember Gandhi's admonition:

    > "Be the change you wish to see in the world"

    And so `pyjanitor` was born.

    Your dataframe manipulation and processing code can now be more expressive than native pandas:

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

    By being the change I wanted to see, Pythonistas now have one more nice thing available to them :).

    ---

    Fast-forward to 2021. I joined Moderna. I had lots of pains with the technology stack at Novartis, and the very forward-thinking Digital leadership at Moderna had a suite of high-power home-grown tools that made it an attractive proposition to join. It was a dog-fooding culture here back in 2021, one which I've fought hard to keep alive within the Digital organization.

    One pain point that I wanted to solve was helping to onboard colleagues onto projects left behind by others. I knew that my work at Novartis was all going to go and die, because NIBR didn't (and to my knowledge still does not) have a common agreed-upon stable way of working that was shared across computational and data scientist divisions.

    Since when I joined, I was only data scientist #6 at Moderna, and since I was hired into a relatively senior role (Principal Data Scientist), I saw the chance to set the standard for Moderna data scientists.

    Together with my wonderful colleague Adrianna Loback (she has since left) and our manager Andrew Giessel (who has also left), we hammered out what the Data Scientists would ship -- a thing called "Compute tasks" (dockerized CLI tools that are run in the cloud) and Python packages -- and designed our entire project initialization workflow around deploying those two things.

    By standardizing on what we ship and then standardizing on the toolchain, we were able to implement a design pattern that made it easy for us to help one another. For the most part, I can, as someone who mostly deals with Research projects, jump into a codebase of a colleague dealing with Clinical Development and be helpful in a modestly short amount of time. (Even faster with AI assistance!)

    A side effect of this is that we were able to design a portable way of working that works best when you give a Moderna data scientist access to a raw linux machine to work on. Paraphrasing what Andrew Giessel once mentioned to me, I came to this realization:

    > Eventually, tools that abstract away the Linux operating system will fail to satisfy users as they grow up and master Linux. They'll want to jump out of a container and just run raw Linux. Anything that tries to abstract away the filesystem, shell scripts, and more eventually runs into edge case, so why not just give people access to a raw linux machine with tools pre-installed? And when we build tools, why not just expose the abstractions in an open source manner?

    And I'm also quite sure you're aware that Tommy is a big fan of the shell too:

    <iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7341052068264128513?collapsed=0" height="265" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

    And so now I'm a big fan of just giving people access to a raw Linux box, outside of a sandboxed container. And it turns out, being able to build a container and run it is a pretty fundamental skill nowadays. So much so that we've basically said "no" to any vendor tooling that forces us to work within a Docker container.

    Here's the most awesome piece of this all: we did this in an "internally open source" fashion. *Anyone* who has a complaint or beef about the tooling can come talk to the Guardians of the Python at Moderna and propose a fix to our tools. Even better, we'll walk you through how to make the fix "the right way", so you gain the superpower of software development along the way!

    <iframe width="560" height="315" src="https://www.youtube.com/embed/3ZTGwcHQfLY?si=_FLzvFyCp88ZlzGm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

    At least on this dimension, we are never going to be beholden to someone else's (or a vendor's) roadmap! We are now _resilient_! Just like how Dustin from Smarter Every Day articulates when he made this video about trying to make a thing "in America".

    I could go on and on about the multitude of design choices we made and upgraded -- to default to the miniforge distribution of conda, to switch to `uv` and `pixi`, to start embracing Marimo notebooks, to move off `setup.py` and to move onto `pyproject.toml`, the upgrade to switch from `handlebars` (a vestige of legacy) to `cookiecutter` -- essentially to evolve the tool stack with the evolution of the Python ecosystem. I will not bore you with them now, but for the curious, I'm more than happy to talk about that. But I'll leave you with this:

    > There's no magic sauce that lies in the tool choices we make. The magic sauce is in the people who choose to show up and build. If your company has these types of people and empowers them to build things that are sensible to build (rather than buy).

    <iframe src="https://www.linkedin.com/embed/feed/update/urn:li:share:7337223460651220992" height="349" width="504" frameborder="0" allowfullscreen="" title="Embedded post"></iframe>

    ---

    Do you remember these beautiful graph diagrams from the beginning of the talk?

    ![](https://ericmjl.github.io/nxviz/examples/arc_node_labels/output_2_1.png)

    There's one more point I want to talk about: building is a great way to learn new things. Building nxviz helped me learn the principles of graph visualization. Building LlamaBot helped me learn about making LLM applications.

    In 2023, I created a new code repo, called LlamaBot. I was confused with how to interact and build with LLMs, particularly RAG applications, and so I decided that I would turn to my favourite learning tool: building software. That turned out to be so clarifying: I was forced to encode my understanding into code, and if the code did unexpected things, I knew my understanding was wrong. After all...

    > Computers are the best student there are. If you teach the computer something wrong, it'll give you back wrong answers. You just have to be good at verification, that's all.

    I built LlamaBot once on top of LangChain at the time. But then I very quickly dropped LangChain as an LLM API provider in favor of LiteLLM. That was my first rewrite. I used it to build CLI tools for myself, like a git commit message writer in 2023, automatic release notes writer, documentation writers, and many more. I then rewrote parts of LlamaBot again, dropping constructs and abstractions in favor of new design choices.

    In total, I've rewritten LlamaBot at least 4 times, each time updating the codebase with the best of my knowledge. Some things that have stayed constant:

    1. The "Bot" analogy, which predates the term "agents", turns out to be a natural way to express Agents.
    2. The docstore abstraction simplifies storage and retrieval for pure text applications.
    3. My distaste for writing commit messages and release notes, and hence the automated writers for both remain deeply ingrained as a dog-fooded tool.

    And some things that have evolved:

    1. QueryBot used to do entire RAG workflows all-in-one -- from PDF->text conversion to embedding -> retrieval. I have since learned that it is much better to break those out into separate steps.
    2. ChatBot used to have a built in ChatUI. I dropped it because it was too opinionated and unwieldy. Marimo has really good chat UI primitives that I think should be used.
    3. Inspiration from the `ell` library: `lmb.user("some prompt")` or `lmb.system("some prompt")` for convenient creation of system and user prompts.

    A lot of the changes were based on updates to my knowledge about what was important to abstract away when working with LLMs, and what was not. Gradually, I'm finding myself building low-level, mid-level, and high-level APIs -- following patterns of the most flexible software that I've seen out there.

    In the process of building and designing software, we have to learn the domain so well that we become essentially a linguistics expert in that domain. Vocabulary, terms, and their relationships become natural extensions of what we already know. If our code maps to the domain properly, we'll find that your abstractions become so natural they're self-documenting. And if our code maps poorly onto a solid understanding of the problem space, as I found out in many occasions, it'll end up being a tangled mess that warrants a rewrite. There's nothing wrong with that! Embrace the need to rewrite; with AI-assistance nowadays, the activation energy barriers to building your own tools is dramatically reduced.

    ---

    <blockquote class="twitter-tweet"><p lang="en" dir="ltr">We are opening up a new role at Quora: a single engineer who will use AI to automate manual work across the company and increase employee productivity. I will work closely with this person. <a href="https://t.co/iKurWS6W7v">pic.twitter.com/iKurWS6W7v</a></p>&mdash; Adam D&#39;Angelo (@adamdangelo) <a href="https://twitter.com/adamdangelo/status/1936504553916309617?ref_src=twsrc%5Etfw">June 21, 2025</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

    ![](https://pbs.twimg.com/media/Gt_VT5nakAANTdj?format=png&name=large)

    Reading this tweet reinforced an important point: Internal tool builds require organizational buy-in. Does your organization empower you to build the tools that you need to get your work done? I was lucky back in the day to have full leadership buy-in through Andrew Giessel and Dave Johnson, and my current manager Wade keeps roadblocks away from innovating on how we work. Nowadays, I also try to encourage that across the teams that I have influence with, even if I don't have direct managerial responsibilities for them.

    ---

    If my arguments don't convince you, then perhaps Hamel Husain, one of the leading AI eval practitioners, will:

    > **Build a custom annotation tool.** This is the single most impactful investment you can make for your AI evaluation workflow. With AI-assisted development tools like Cursor or Lovable, you can build a tailored interface in hours. I often find that teams with custom annotation tools iterate ~10x faster.
    >
    > Custom tools excel because:
    >
    > - They show all your context from multiple systems in one place
    > - They can render your data in a product specific way (images, widgets, markdown, buttons, etc.)
    > - They’re designed for your specific workflow (custom filters, sorting, progress bars, etc.)
    >
    > Off-the-shelf tools may be justified when you need to coordinate dozens of distributed annotators with enterprise access controls. Even then, many teams find the configuration overhead and limitations aren’t worth it.
    >
    > [Isaac’s Anki flashcard annotation app](https://youtu.be/fA4pe9bE0LY) shows the power of custom tools—handling 400+ results per query with keyboard navigation and domain-specific evaluation criteria that would be nearly impossible to configure in a generic tool.

    And he makes a great point:

    > With AI-assisted development tools like Cursor or Lovable, you can build a tailored interface in hours.

    The barrier to entry for building your own tools nowadays is so much lower than it was before. A lot of the grunt work for building your own tools can be automated away using a combination of templating and LLM assistance.

    ---

    I love the work that I do because I have an outlet for expressing my creativity through the tools that I make for myself and others to use. Through my nearly 10 years of making tools for myself, I've crystallized this as a lesson in scaling.

    1. Software scales our labour.
    2. Documentation scales our brains.
    3. Tests scale others trust in our code.
    4. Design scales our agility.
    5. Agents scale our processes.
    6. Open source scales opportunity for impact.

    If you can build software tools for yourself, you can scale yourself. If you teach others to use those same tools, you can scale their labour. And you can scale your brain by documenting those tools well. If you test those tools thoroughly, you can scale the trust in the codebase, enabling others to contribute with confidence. If you design the software well, and more importantly, design the business process that software supports well, you can become nimble and agile without the trappings of Big Fake Agile. If you use agents as part of the custom tooling, you can scale those same processes even further. And if you make your tooling open source (whether internally or externally), you scale the opportunity for others to contribute.

    And so, my fellow builders, let's build. Not because your company wants it of you, but because patients are waiting.

    > There are people dying, if you care enough for the living, make a better place for you and for me.
    >
    > -- Heal the World (Michael Jackson)
    """  # noqa: E501
    return (talk_source,)


@app.cell
def _(deckbot, mo, talk_source):
    deck = deckbot(talk_source)
    mo.md(deck.render())
    return (deck,)


@app.cell
def _(Path, deck):
    deck.save(Path("index.md"))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(openai, os, tempfile):
    def transcribe(microphone):
        # Save the audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(
                microphone.value.getvalue()
            )  # Use getvalue() to get bytes from BytesIO
            temp_file_path = temp_file.name

        try:
            # Use OpenAI's Whisper API to transcribe the audio
            client = openai.OpenAI()  # Assumes API key is set in environment variables

            with open(temp_file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )

            # Display the transcription
            return transcript

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return (transcribe,)


@app.cell
def _(deckbot, microphone, mo, transcribe):
    transcribed_deck = None
    if microphone.value:
        transcribed_deck = deckbot(transcribe(microphone).text)
    mo.md(transcribed_deck.render())
    return (transcribed_deck,)


@app.cell
def _(mo, transcribed_deck):
    edit_microphone = mo.ui.microphone(label="What would you like to edit?")
    slide_selector = mo.ui.dropdown(
        label="Select the slide you want to edit.",
        options=list(range(len(transcribed_deck.slides))),
    )
    mo.vstack([slide_selector, edit_microphone])
    return edit_microphone, slide_selector


@app.cell
def _(edit_microphone, mo):
    mo.audio(edit_microphone.value)
    return


@app.cell
def _(edit_microphone, transcribe):
    transcribed_edit_request = transcribe(edit_microphone).text
    transcribed_edit_request
    return (transcribed_edit_request,)


@app.cell
def _(deck, mo, slide_selector, transcribed_deck, transcribed_edit_request):
    transcribed_deck.edit(slide_selector.value, transcribed_edit_request)
    mo.md(deck.render())
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
