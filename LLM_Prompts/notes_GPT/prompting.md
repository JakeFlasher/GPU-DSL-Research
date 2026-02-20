## Tips on Research
1. when accessing arxiv papers (including huggingface.co/papers/<arxiv_id>), always use arxiv/html/<paper_id>
2. proposals generation are easily accessible and generous; need a decent evaluator for proposals
## Tips on Code Gen
1. Use Linus's code style
2. Xml format, json banned
3. Directly embed the source code implementation when debugging
## be as specific as possible
1. revision, divide each review, each section
2. tailor the prompt to each individual review
3. always use multi-stage, refer to OpenAI/Anthropic official guide
4. always provide the exactly needed source files (work on structure first, use large context window models to detect the needed files, then use intelligent like GPT 5.2 pro to tailor and modify the details)
