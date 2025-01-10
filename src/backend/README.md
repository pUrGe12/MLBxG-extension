# Backend

This is where the processing happens. The important files of consideration are:

1. The [main file](./main.py)
2. [API querying files](./API_querying)
3. The helper models [part 1](./models)
4. The helper files [part 2](./helper_files)

Additionally we have the [uploads file](./uploads/videos) which is where the user's inputted videos (as will be explained later) go.

## Functionality

The backend performs the following functions

### Panel

1. LLM intergration that answers basic baseball questions
2. API query pipeline for the LLM to answer questions based on player's and team's information as well as the schedule for MLB
3. Generate statcast data for a live video being watched on YouTube (this is a bit iffy)

For example:

The user can ask questions like "how fast was that ball?" and the LLM will be able to answer that! 

### Options

1. Using pinecone to store home-run stats for top MLB players and comparing the user's stats with them to let them know how good or bad they are (through an LLM again)
2. There is a guide that explains the techincal workings of the extension (much like this page itself)
3. User's can enter a classical MLB match directly (in mp4 format) or enter the name of the match (and we'll scrape for ourselves) to generate it's statcast data 
(as data recorders were not there)

## How we do it?

The complete techincal explanations are here:

### Panel

1. 