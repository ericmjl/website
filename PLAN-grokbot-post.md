# Plan: Grokbot blog post

Planning only. A later run writes the post from this file. Do not treat this file as the post, and do not add `content/blog/` yet.

The post is first person, for Eric Ma's Lektor site. Source of truth is his 11 November 2026 voice recording (about 32 minutes). The auto-transcript is `uploads/transcript.txt` in the planning workspace. Use only what that transcript says. Invent no facts, numbers, links, or quotes. Leave out other people's private details.

## Proposed title

Just talking to Grokbot

Sentence case, no colon. It uses his closing phrase, "the simplicity of just talking with it."

Alternate, if he rejects that title: Grokbot made personal agents easy. That one tracks the opening claim that it "turned personal agents into something really darn easy." Do not invent a third title.

## Slug

`just-talking-to-grokbot`

No existing `content/blog/` slug matches. The later contents file is `content/blog/just-talking-to-grokbot/contents.lr`. Public URL, once published: `/blog/2026/11/13/just-talking-to-grokbot/`.

## Date

`pub_date: 2026-11-13`

The recording is from 11 November 2026. The post date is 13 November 2026. Do not narrate the recording session, and do not write "yesterday" or "today" off either date.

## What the post has to do

His thesis, from the end of the recording, is the close, not the lede. The simplicity of just talking to it, plus very easy connectivity, sets a pattern for working with AI agents. The opening still has to put the billionaire caveat before any praise.

Cover every thread below. If the draft feels like a feature tour, merge adjacent sections. Do not drop a thread.

1. The billionaire-owner caveat, up front.
2. The chat-only screen, with the configuration hidden, set against pi, T3 Code, and raw Cursor.
3. About 25 connected services.
4. A chief-of-staff bot, plus specialized bots that consult one another.
5. Coding delegated to Cursor cloud agents, with screenshots or links as proof.
6. A background desktop he can reach from his phone.
7. Notifications, short mobile replies, and hands-free voice.
8. A once-a-day personal email check that drafts replies.
9. Personal bots: the bike warranty bot and the chef bot.
10. Starting simple and building trust.
11. The cost caveat.

## Field skeleton the writer will use

Match recent posts (`prompt-cards-teach-the-why-behind-the-words`, `a-little-tool-for-sharing-prompts` on `cursor/glosscard-blog-post-3dfb`). Each field ends with a `---` line of its own.

```
title: Just talking to Grokbot
---
author: Eric J. Ma
---
body:

(prose)
---
pub_date: 2026-11-13
---
twitter_handle: ericmjl
---
```

Do not hand-write `summary:`, `tags:`, or `logo.webp`. Those come from the blogbot skill after the prose passes. See the checklist.

Headers inside `body` are `##` only, sentence case, no colons, no `###` unless a section is actually doing two jobs. This post is a narrative. Prefer no subheads.

## Sources for voice and format

Read before drafting:

- This plan, then the transcript, in that order. The transcript wins on facts.
- `AGENTS.md` in this repo: voice habits, "Language to Avoid," Lektor field separators, blog image rules, PR title rule.
- `.agents/skills/blogbot/SKILL.md`, for what happens after the prose is done. Summary, tags, and banner only.
- Recent posts for rhythm and `contents.lr` shape:
  - `content/blog/prompt-cards-teach-the-why-behind-the-words/contents.lr` (2026-10-23)
  - `content/blog/four-pillar-obsidian-knowledge-wiki/contents.lr` (2026-10-22)
  - `content/blog/live-demoing-data-science-in-the-age-of-agents/contents.lr` (2026-10-14)
  - `content/blog/plaud-note-pro-review/contents.lr` (2026-09-20), a short product review
  - GlossCard draft on branch `cursor/glosscard-blog-post-3dfb` (PR 330): `content/blog/a-little-tool-for-sharing-prompts/contents.lr`, `pub_date: 2026-11-12`. Closest sibling: a small tool, first person, concrete, no slogan close.

`ghostwriting-blog-post` and `references/language-to-avoid.md` are not in this checkout and not on `origin/main` (code search returned nothing). The same avoid-list is already in `AGENTS.md` and in the vault writing rules. Follow those. Do not go hunt a missing skill and stall.

Voice samples used while planning (do not quote them into this post): "Make AI your adversary" and "Canvas Chat: The sprint that became a marathon," plus the GlossCard voice-pass notes. Relevant habits from that pass: name a person as the actor, break slogan-y contrasts into plain sentences, little or no bold in body prose.

## Name and hearing corrections

Spell the product **Grokbot**, one word, the way he says it. He never says "Grok Bot" in the recording. Grubbot, Grovbot, Brockbot, Grockbot, and "crockpot" are transcript errors for the same product. This spelling is an open question. Do not switch it quietly.

| Heard in the transcript | Use in the post | Notes |
| --- | --- | --- |
| Grubbot, Grovbot, Brockbot, Grockbot, crockpot | Grokbot | Open question. See below. |
| Py, "a harness like Py" | pi | He is comparing chat to a coding harness. Likely the pi harness. Open question. Do not describe pi beyond "a harness where the model and the other settings are in front of you." |
| T3 code, T three code | T3 Code | He has recently started using it. It shows more configuration than Grokbot. Later, its remote mode works while his Mac is online and lacks Grokbot's computer isolation. Both roles are real. Keep them distinct. |
| Raw Cursor | Cursor, the app he configures by hand | The contrast is the busy surface, not a rejection of Cursor. He still sends coding to Cursor cloud agents. |
| Versal | Vercel | Hearing correction. Website host for the expense app, and one of the connections. |
| thin bases | Do not use | He already said Convex in that sentence, and later "Convex for the database." Do not add a second datastore. |
| One Password | 1Password | Hearing normalization. Open question. It is in the connection list. |
| Electric, Electrics, "L E C T R I C" | Lectric | He spelled the letters. The e-bike brand in the warranty story. |
| OpenBot, "by the Copilot Kit group" | Keep his words | Do not rename it, do not add a URL, do not decide it lacks computer isolation. Open question. |
| Ghost GhostPal, Ali Abdal | Omit | Product name and spelling are too uncertain. The hands-free point does not need this comparison. |
| Hey, "bubbling feature" | Only what he describes | Do not import HEY's documentation. Open question. |
| impeccable skills | Omit as a connection | Said once, right after 1Password. Do not add Impeccable to the 25. |
| Rough box, Contacts window | Do not quote | Recoverable meaning only: Cursor agent work stays out of the Grokbot chat, and he likes that. |

## Section outline

1. I know who owns it
2. You only see the chat
3. Start with something small
4. About twenty-five connections
5. A chief of staff and a bot per thing
6. The coding runs on Cursor
7. A desktop in the background
8. Short enough for a phone
9. Personal email once a day
10. A bot for the bike repair
11. The sixty-dollar plan gives me pause
12. Just talking, and easy connections

Chef bot lives in section 8, because he told it while describing hands-free cooking. Bike bot is section 10. Both are required.

## Key points by section

Timestamps are the transcript's. Paraphrase into written sentences. Clean stammers. Do not upgrade his diction, and do not add a detail that has no timestamp.

### 1. I know who owns it

- [00:00] to [00:31]. It has been "all the rage." He has his own issues with billionaires buying elections and the like. He knows this is a product by a company owned by a billionaire. He wants to set that aside for a moment.
- [00:31]. The merit he is willing to talk about: it turned personal agents into something really darn easy.
- [01:05] to [01:17]. The real power is simplicity. He gives that to the team. He did not appreciate simplicity until Grokbot.
- Do not name the owner, the company, or any election. He didn't.

### 2. You only see the chat

- [01:39] to [02:12]. He is a coder. He is used to environment variables and hand-tweaking models. Doing that much gets tiring. A bit of simplicity is refreshing. He stumbles toward this sentence; write it once, cleanly.
- [02:39] to [03:00]. In the chat you do not see much. You see the chat window. You talk to a bot. That is all it is.
- [03:01] to [03:35]. Compared with a harness like pi, with T3 Code (he has recently started using it), or with raw Cursor. Those put more options in front of him immediately. He can configure the model and configure other stuff. Grokbot hides that. He starts chatting as if he were talking with another human.
- [03:35] to [03:55]. He finds that really cool, and extremely simplifying. One less thing to worry about. His reading of the team's bet: you just want to accomplish your stuff. Attribute that to him, not as a quote from the team.
- The chief-of-staff naming convention is spoken here ([00:47] to [00:58]). Save it for section 5. One clause is enough if the open needs it.

### 3. Start with something small

This is the trust thread. He rewound to put it before the organization story.

- [05:49] to [06:12]. Once you have the first bot, start simple. Chat and get used to it. Hand over tasks you are comfortable with. He first says he was comfortable with it reading his email.
- [06:12] to [06:24]. Then the caveat, in his words: as a consumer product, trusting that companies will not use our data is too naive. Way too naive. Do not soften this into "I trust them" and do not expand it into a privacy essay.
- [06:24] to [07:13]. He corrects the starting point. Calendaring was one of the first things, because he is too lazy to create a calendar invite. He takes a photo of something he needs to remember, sends it, and says "Make a calendar invite for me." It does the right thing.
- Write the correction. Email is not the story he landed on for "first task." The daily email routine is section 9, a later, heavier workflow.
- Dictating a calendar event to the chief of staff ([14:38] to [15:12]) belongs in section 5. Do not tell the photo story twice.

### 4. About twenty-five connections

- [04:03] to [04:26]. He gets things done by asking for the connections. He names Gmail and Google Calendar first, then says he has connected it to other things for personal projects.
- [04:26] to [04:31]. "I have twenty five connected. So many of them."
- [04:31] to [05:07]. Names he actually lists. Deduped, he says Google Calendar, Convex, Stripe, Notion, Calendly, Google Docs, Sheets, Slides, Buffer (he does social media stuff), Mercury (he says "Bank, Mercury"), Cloudflare, Resend, Vercel, GitHub, Gmail, and 1Password. Calendly, Gmail, and Google Calendar come up more than once. Do not double-count them.
- That named set is shorter than 25. Say he counted about 25, then name the ones he listed. Do not invent the rest. Do not add Modal, Cursor, or Impeccable to this inventory. Modal and Cursor show up later for different reasons.
- [04:31]. Convex is in the list "because I do a lot of my personal projects with" what the transcript hears as "thin bases." Treat that phrase as a garble. The fact to keep: Convex is what he uses for a lot of personal projects. Section 6 also says Convex is the database for the expense app.
- [05:07]. "Those are all the impeccable skills" is not a clear extra product. Leave it out.
- [05:12] to [05:22]. He connects them and can suddenly do a whole boatload of stuff. He calls it really empowering. Keep "boatload" if it still sounds like him in prose. Do not replace it with "a wide range of workflows."

### 5. A chief of staff and a bot per thing

- [00:47] to [00:58]. The convention he follows is to call the first bot chief of staff and let it start doing things.
- [14:34] to [15:24]. Chief of staff is where he dumps everything. A new calendar event: he dictates it to the chief of staff. An email to draft: he tells the chief of staff. One bot is one thing in his life he needs to keep track of. The chief of staff is the dumping ground for everything else. More like a secretary or assistant. "Chief of staff is a nice name for that. A glorified assistant."
- [11:26] to [11:56]. Bot-to-bot communication: he does not leverage it very much. He knows it is a cool feature and has used it a few times. The example: a chief of staff, and another bot specialized for his personal email that knows some of the quirks of how he organizes that mail. When the chief of staff needs to know something about that mail, it goes to the personal email bot. Do not turn "a few times" into a society of bots consulting each other all day. The user asked for this thread; the honest version is the email handoff, plus the admission that he rarely does it.
- [11:56] to [12:18]. He can group bots. He has bots for the Learn Anything Retreat. His open source projects have one bot per project, and those bots dispatch Cursor agents for each pull request. Do not name the projects. Do not import retreat dates, co-teachers, or URLs from the blogbot skill. None of that is in the transcript.
- [12:18] to [12:48]. He also has bots whose job is to monitor something he does not want to forget. The bike is the example. Point forward; tell the story in section 10.
- [25:29] to [25:44]. He does not use the Slack connection. He does not think he needs it with teams of people just yet. One sentence, optional, so the post stays about personal work. Not a section, and not a feature matrix.

### 6. The coding runs on Cursor

- [07:17] to [08:05]. Grokbot is connected to his Cursor account, so he can create Cursor agents from the chat. He says "Spin up the cursor agent," and it does it. No clicking around. The same request can be spoken. Ignore "crockpot" at [08:05]. It means Grokbot, and the voice detail belongs in section 8.
- [07:34] to [07:53]. "I can have my Cursor agents dispatch off Cursor agents" is muddy. Do not build an agent hierarchy out of it. The clear instruction is [09:19] to [09:29].
- [08:16] to [09:19]. Example: his own expense app. He says he is securing it in a certain way, and does not say how. Do not invent auth. Vercel for the website, Convex for the database, Modal for some on-demand AI features, Resend for email in and out. One bot is dedicated to that app. He tells it what he wants. He has to keep the discipline of one feature at a time so he does not overwhelm the bot. He also gave it an instruction.
- [09:19] to [09:29]. That bot is not supposed to handle the coding, because coding consumes his Grokbot allowance very quickly. It is allowed to spin up subagents via Cursor cloud agents.
- [09:29] to [10:22]. His laptop can be off. He is on his phone. He has instructed Grokbot that it must prove the work to his satisfaction by sending screenshots, ideally, or links where those are the relevant proof. The screenshots show him quickly when something is off. He then says what he wants, and the work continues in the background while he goes about his life.
- [10:22] to [11:06]. He likes this. Notifications on his phone and on his watch let him fire and forget until his brain is in the right state. If it is not, he needs the discipline to say no and not look at his phone. Stuff happens in the background either way.
- [16:00] to [16:16]. Dispatching to a Cursor agent works well. Grokbot chooses the model that the Cursor agent will use. He does not name the model. Do not name one.
- [16:28] to [16:49]. Cursor agent work does not clutter the Grokbot conversation. He is glad. Do not quote "Rough box" or "Contacts window."

Keep two money facts apart. This section: coding inside Grokbot burns the Grokbot allowance quickly, so he pushes coding to Cursor. Section 11: a separate, recent change in limits, measured as Cursor credits. Do not merge them into one theory of billing.

### 7. A desktop in the background

- [16:49] to [17:16]. When Grokbot's computer is running, he can take control. He hit a wart and talked through it live.
- [17:08] to [17:51]. He thought he could not paste from his own computer into Grokbot's computer. He took control, opened Chrome, copied "Hello world!" from Spotlight, and the first paste looked wrong. He cut the text instead and confirmed that he can paste. The post uses the correction: he can copy or cut on his machine and paste into Grokbot's computer. Do not write this up as a paste bug. The Hello World test can be one short beat or omitted, as long as the corrected result stays.
- [18:01] to [18:31]. Why he takes control: he does not want passwords sitting with the bot, or Grokbot is having trouble with its secure password form. That form does not store his password. It passes the password through into the UI. He can also take control from his phone. He calls it almost a cloud computer, a fully fledged desktop that runs in the background.
- [18:37] to [18:47]. That computer is how Grokbot takes screenshots. The Cursor agent can take screenshots too.
- [18:58] to [19:35]. He pauses and asks whether something was scheduled for December. Leave the whole aside out. Do not invent an event.

### 8. Short enough for a phone

Notifications and "fire and forget" are already in section 6. This section is how the replies themselves have to work on a phone, including voice. The chef bot is the cooking example.

- [19:38] to [20:18]. What he likes about the interaction: he can keep his energy on where the top-level stuff is. If he forgets something, he types it and gets a quick confirmation. The Grokbot models are fast, so they respond quickly. No model names.
- [20:20] to [21:31]. Tuning. The chief of staff, or whatever he calls the main bot, needs to be able to coordinate. The bots need instructions to answer in a shortened, concise way. Without that, Grokbot likes to be very verbose. As soon as he sees walls of text, especially on his phone, he tells that bot: be more concise. Messages have to be easy to view on a phone. Do not give long walls of text. If a long answer is actually necessary, offer to walk through it one item at a time.
- [21:37] to [22:12]. That prompt makes the replies easier to review. It is also easier to swipe right and reply to one issue, comment, or message, and deal with it at that grain.
- [26:00] to [26:11]. He can have hands-free voice conversations. He has done this while biking, while scootering, and while cooking.
- [26:11] to [27:21]. He made a chef bot for cooking. If he needs proportions, it is instructed to rely on the best reliable information. How much salt, how much to weigh, portion sizes. He can also store feedback, including that a recipe it found was a dud, and have it remember. The point is not that it is perfect the first time. It is that the bot becomes a learning system that adjusts to his preferences.
- [27:21] to [27:44]. Hands-free, it can still call tools, including coding agents. He has been on calls of five minutes and of ten minutes to hash something out.
- [27:54] to [28:23]. He mentions a specialized voice app, heard as "Ghost GhostPal" by "Ali Abdal and Co.," which he loved at launch and no longer uses because Grokbot and other agents he is testing already do that job. Omit this. The name is too uncertain to print, and a vague substitute would be an invented claim.

### 9. Personal email once a day

- [22:18] to [22:41]. A short bridge is enough: he has used it to fill out forms, coordinate coding tasks, and draft emails. Stop there. "I've used it to send" dies unfinished. Do not complete it.
- Omit [22:27] to [22:32], registering his kid for skating class. That is someone else's private detail. Do not replace it with a made-up form.
- [22:41] to [23:21]. He has a personal-email routine. He looked, and it still checks every day at noon. He forgets the exact setup because he says what he wants in natural language and Grokbot creates the automation.
- [23:21] to [23:46]. It checks his email at noon, prepares drafts, and bubbles them up. He is using "the bubbling feature in Hey." Anything bubbled up is something he needs to answer. If he has replied, it comes out of the bubble. Describe only that behavior. Do not explain Hey's product.
- [23:49] to [24:12]. He thinks the check should move to 11 a.m. He has not said that he changed it. At lunch he is usually busy and has no time to check personal email, let alone answer it. The draft is a first pass. He edits, and he has the bot learn from that.
- [24:29] to [24:51]. He says he really only checks personal email once a day now, then corrects himself: that is an ideal state he is moving toward. Use the correction. He knows he cannot triage every email. He has to use AI agents as a buffer. There is too much information.

The chief-of-staff "draft this email" request in section 5 is ad hoc. This section is the daily pass. Do not collapse them.

### 10. A bot for the bike repair

- [12:18] to [13:00]. The rear wheel on his bike is broken. It is a long-running, low-humming, back-burner chore, and it is stressful to keep hold of. He has a bike repair bot. Its lifespan is the lifespan of the repair.
- [13:00] to [13:22]. He was requesting a warranty replacement of the rear wheel of his Lectric electric bike. He spells L E C T R I C. The wheel broke after it had already been replaced. He wanted a warranty replacement of that replacement.
- [13:22] to [13:42]. Lectric's policies do not allow that. They did discount a rear wheel, and he was happy to buy it that way. The bot drafted the emails, so he did not have to do that part under stress.
- [13:42] to [14:06]. He put a rule in the bot: if Lectric does not respond within two business days, follow up. He does not want to remember that kind of detail. Grokbot can set a recurring reminder and can update its own reminders as he steers it.
- [14:06] to [14:20]. This is the trust payoff, said generally: he delegates the boring, dumb little things he does not want to do or talk about. That takes stress off his life, and he ends up in a better position. "It's very confidence building." Tie it back to section 3 without repeating the calendar photo.

### 11. The sixty-dollar plan gives me pause

- [28:23] to [28:48]. What is not good right now: the limits changed recently. He does not say which limit, or why. In three days he burned through 30 percent of his Cursor credits. Those credits are combined with Grokbot. He calls that ridiculous.
- [28:48] to [29:05]. That pace used to be what he would spend with Claude. He does not say over what period, or on what Claude plan. Do not invent either. He is on the sixty-dollar-a-month plan.
- [29:05] to [29:36]. Something changed about the economics. He is not sure what. It is one reason he is considering deploying OpenBot for himself. He says OpenBot is running on his Mac right now, "by the Copilot Kit group." He is also looking at other ways of doing remote work. Do not say OpenBot lacks computer isolation. He does not say that.
- [29:36] to [30:00]. T3 Code can do remote work as long as his Mac is online. It does not have the same computer isolation he is used to with Grokbot. That tradeoff is the one he says he has to keep in mind.
- [30:04] to [30:50]. Set against that, the first touch was simple, and he could grow into it. That part was really cool. The only thing giving him pause is that the unit economics of token spend have fundamentally changed on the more expensive plans. He used to be very comfortable inside the sixty-dollar plan. Now not so much. It is making him do a double take. He wants that on the record.
- Do not convert 30 percent into dollars. Do not add a Grokbot subscription price. He never names one.

### 12. Just talking, and easy connections

- [31:16] to [31:29]. He says he will turn the recording into a blog post. Leave that out.
- [31:29] to [31:53]. Grokbot is setting a pattern for how to work with AI agents. The two things he wanted to highlight: the simplicity of just talking with it, and the power of very easy connectivity.
- End on those two. Do not add a third moral. Do not recap the bike, the bill, and the desktop in a closer. The cost section already gave the pause. This section is the pattern he still thinks is worth copying.

## Phrasings to keep

Clean repeated words and false starts. Keep the attitude. These are speech, not lines to drop in with quotation marks unless a line is already a thing he said to the bot. "Make a calendar invite for me" is one of those.

- "I have my own issues with billionaires buying elections and the likes." [00:06]
- "this is a product by a company owned by a billionaire." [00:15]
- "set aside that for a moment." [00:29]
- "turned personal agents into something really darn easy." [00:31]
- "The real power actually comes from simplicity." [01:05]
- "I did not appreciate it until Grokbot." [01:17]
- "I really have to give it to the team." [01:05]
- "I am a coder." Environment variables, hand-tweaking models, and it gets tiring. [01:39] to [01:54]
- "refreshing." [02:07]
- "You really don't see much. You just see the chat window." [02:50]
- "You just talked with a bot. That's all it is." [02:58]
- "as if you were talking with another human." [03:11]
- "It's extremely simplifying. There's one less thing to worry about." [03:39]
- "you just want to accomplish your stuff." [03:55]
- "I have twenty five connected." [04:26]
- "a whole boatload of stuff." "It's really empowering." [05:12] to [05:19]
- "you gotta start simple." [05:54]
- "too naive." On trusting companies not to use the data. [06:24]
- "I'm too lazy to create a calendar invite." [06:24]
- "Make a calendar invite for me." [06:57]
- "Spin up the cursor agent." "There's no clicking around." [07:53] to [07:59]
- "one feature at a time." [08:59]
- "that really consumes my Grokbot allowance very quickly." [09:19]
- "my laptop can be off and I'm on my phone." [09:29]
- "prove that it's done the work to my satisfaction by sending me screenshots, ideally, or links." [09:29]
- "fire and forget." The discipline to not look at the phone. [10:33] to [10:47]
- "a glorified assistant." Dumping ground. One bot, one thing in his life. [15:12] to [15:24]
- "I don't leverage that very much." Bot to bot. [11:26]
- "boring, dumb little things." "taking stress off your life." [14:06] to [14:20]
- "If Electric does not respond within two business days, I'm following up." Print Lectric, not Electric. [13:42]
- "walls of text." "You gotta be more concise." "easily viewable on a mobile phone." Walk through one by one if a long answer is unavoidable. [20:49] to [21:31]
- "a fully fledged desktop computer that runs in the background." "almost like having a cloud computer." [18:23] to [18:31]
- "doesn't store my password but just passes through into the UI." [18:01]
- "not that it gets it perfect from the first time" but that it "becomes a learning and adaptive system." [26:59]
- "five-minute, ten-minute calls." [27:21]
- "I really only check my personal email once a day now." Then the correction that this is a state he is moving toward. [24:29] to [24:36]
- "use AI agents as a buffer." "too much info." [24:40] to [24:51]
- "Thirty percent of my cursor credits." "in three days." "ridiculous." "sixty a month." [28:34] to [28:48]
- "the unit economics of token spend have fundamentally changed." "double take." [30:10] to [30:42]
- "the same kind of computer isolation." About T3 Code's remote mode, not about OpenBot. [29:36]
- "The simplicity of just talking with it, coupled with the power of very easy connectivity." [31:39] to [31:42]
- "a cool pattern for how to work with AI agents." [31:29]

Use "really cool" at most once. He says it many times. In prose, once is a punch and three times is a tic.

## Leave out

- His kid, and skating class. [22:27]
- The December "did we schedule" pause. [19:09] to [19:35]
- "I'm going to turn this into a blog post" and "we'll get the bots to draft a blog post." [31:16] to [31:53]
- Hello World, except as a one-line record of the paste correction if the section needs it. No transcript stage directions.
- Slack, unless it is the single optional sentence in section 5.
- The Ali Abdaal voice-app comparison. [27:54] to [28:23]
- "Impeccable skills" as a connected service. [05:07]
- "Thin bases" as a product. [04:31]
- Any URL. He did not speak one. Do not add product homepages. A homepage would smuggle in claims from the page.
- Retreat logistics that live in the blogbot skill and not in the transcript.
- Dollar math he did not do. No second subscription price. No Claude invoice.
- Other people's names, except Lectric as a company he named by spelling it. He names no private person we should print. "Ali Abdal" is omitted with the product-name problem, not because the person is private.

## Open questions

The writer does not resolve these by guessing. Defaults below are so the draft can proceed. Eric can override any of them.

1. **Spelling of the product.** Default: Grokbot, because that is how he says it and this plan was told to spell it that way. He may want "Grok Bot" or another official form. Flagged, not settled.
2. **pi.** Default: "a harness like pi," with no link and no feature list. Confirm the name before anyone links it.
3. **1Password.** Default: 1Password. The transcript says "One Password."
4. **OpenBot and "the Copilot Kit group."** Default: one sentence in his words, no link, no official respelling, no claim that it lacks isolation. He says it is running on his Mac and that he is considering deploying it for himself.
5. **Computer isolation.** Attach that phrase to T3 Code's remote-while-the-Mac-is-on setup only.
6. **"Hey" and bubbling.** Default: describe the behavior he described (bubbled means he needs to answer; a reply takes it out of the bubble). Do not decide that Hey is a specific email product in print unless he confirms. Do not describe Gmail as if it were the bubbling feature.
7. **Lectric.** Default: Lectric, from the letters he spelled. Do not write "Electric."
8. **Paste.** Default: he can paste. The early "I can't paste" was him mid-test.
9. **Email cadence.** Default: the automation still runs at noon. Moving it to 11 a.m. is what he thinks he should do. Once a day is the state he is moving toward, not a finished habit he is bragging about.
10. **The unnamed nine-or-so connections.** Default: "about 25" plus the names he listed. Do not pad.
11. **Title.** Default: Just talking to Grokbot. Alternate is in the title section.
12. **GhostPal / Ali Abdaal.** Default: omit.

## Post-writing checklist

Do these in order. Do not skip ahead to blogbot.

### Draft

- Create `content/blog/just-talking-to-grokbot/contents.lr` with `title`, `author`, `body`, `pub_date`, and `twitter_handle` only.
- Prose, first person, active. No bullets with bold headers, no emoji, no bold sprinkled through the body.
- Sentence-case `##` headers, no colons in headers.
- No em dashes, no en dashes, no arrows. Commas, periods, semicolons, or a new sentence.
- Straight quotes.
- Every concrete claim in the draft should be traceable to a timestamp in this plan. If it is not, cut it.

### Pass 1: coherence

Read the draft once as a reader who has not heard the recording.

- The billionaire caveat is the first move, before any praise.
- The two-part thesis arrives at the end, in section 12. The opening may say personal agents got easy. It may not already deliver "just talking, plus connectivity" as a neat bow.
- All 11 threads in "What the post has to do" are present. Chef and bike both appear.
- No repeated scene: the photo-to-calendar story once, the noon email routine once, screenshots once.
- Self-corrections landed on the corrected fact: paste works; calendaring was the early trusted task; once-a-day email is a direction, not a finished victory; bot-to-bot is rare; the check is still at noon.
- T3 Code's two appearances still mean two different things (a busier configuration surface, then a remote mode without Grokbot's computer isolation).
- The allowance remark in the coding section and the 30 percent figure in the cost section are not fused.
- "About 25" is not presented as the length of the named list.
- Nothing in "Leave out" slipped back in.
- No new number, product, URL, or person.
- Section order matches this plan, or a merge is noted and no thread was lost.
- The last paragraph can stand alone as his point. It does not start with a summary phrase.

### Pass 2: active voice

Go sentence by sentence. The grammatical subject should be a person or a named actor: he, the chief of staff, the bike bot, Lectric, the Cursor agent. From the GlossCard voice notes: if a sentence hides who did it, name them.

Prefer:

- "I connected Gmail" over "Gmail was connected."
- "Grokbot hides the model picker" over "the complexity is hidden away," unless the next sentence already says what he does instead.
- "I tell it to send screenshots" over "screenshots are requested."
- "Lectric discounted the wheel" over "a discount was offered."
- "I take control from my phone" over "control can be taken."

Passives that can stay, and why:

- "That form does not store my password" describes the form's behavior he stated. An extra actor adds nothing.
- "It is stressful to keep hold of" can stay if the sentence already says the chore is on his back burner. Otherwise write "I do not want to keep holding it."

Do not flip a sentence into active voice by anthropomorphizing an object. "The chat window decided" is worse than a passive. He can say the team bet on simplicity, because he said he gives the team credit.

### Pass 3: write like Eric

Do this pass after coherence and active voice, not before. Re-read `AGENTS.md` voice habits, the PLAUD review, and the GlossCard post. Then edit the draft against this list.

- He is the subject of his own choices. "I assumed," "I told it," "I was too lazy." Not "the setup assumed" and not "the model suggested."
- Make the consequence explicit when a fact needs one. He pushes coding out to Cursor because doing it inside Grokbot burns the allowance. The screenshots matter because he can see, fast, that something is off.
- Short paragraphs. A takeaway gets its own sentence. An exclamation only where he actually punched ("really darn easy," maybe one "really cool").
- "Here is an example" uncontracted, if a narrative example needs a lead-in. He uses contractions everywhere else (I'm, don't, that's).
- Semicolons only to join two short clauses that are one thought.
- A parenthetical aside has to be specific and optional. Skip wry asides that the transcript did not earn.
- Keep his plain words where this plan lists them: darn easy, boatload, glorified assistant, boring dumb little things, double take, too naive, too lazy.
- Cut anything in the `AGENTS.md` avoid-list: "stands as," "it's important to note," "in summary," "not only but also," rule-of-three closers, "as of [date]," empty "some argue."
- No letter greeting, no "I hope this helps," no meta about being a transcript.
- Read the draft aloud. If a sentence could only have been written by a language model smoothing a speech, break it back into the way the PLAUD post and the GlossCard post move: a claim, the concrete thing he did, what that meant for him.

### After the three passes: tags, summary, and banner

The writer does not hand-write tags, summary, or the banner. They come from the blogbot skill afterwards.

From inside the worktree, run these one at a time, never in parallel (the scripts share one sqlite file and collide):

```
uv run .agents/skills/blogbot/scripts/summary.py just-talking-to-grokbot
uv run .agents/skills/blogbot/scripts/tags.py just-talking-to-grokbot
uv run .agents/skills/blogbot/scripts/banner.py just-talking-to-grokbot
```

Then:

- Scrub the summary before it goes into `contents.lr`. No `[URL]`. No "Read on" tail. No em dash. It is a first-person meta description, and the template already adds the read-more link.
- Read every tag. Fix duplicated letters and fake compounds before saving. Tags are short strings, at most two words, and the script aims for 10, with at least 7 of them one word.
- The banner script writes `logo.webp` into the post directory. Do not draw a substitute by hand.
- If a script falls back in a way the blogbot skill marks as broken, follow that skill. Do not paste a hand-written summary to get around it.

### Build and the later PR

- Build with `pixi run build` from this repo. Fix Lektor parse errors. A missing `---` between fields is the usual cause.
- This planning run does not open a PR. When a later run does, the title needs the publication date at the end, from `AGENTS.md`: `blog: add Just talking to Grokbot [2026-11-13]`. If `pub_date` changes, the PR title changes with it.
- No inline images are required. He did not describe a figure. If the later run adds one, it is a `.webp` beside `contents.lr`, kebab-case, with a real alt string. Do not invent a screenshot he did not ask to publish.
