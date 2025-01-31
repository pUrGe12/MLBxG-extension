<div align="center"><a name="readme-top"></a>

<img src="./images/icon.png" alt="Icon" style="width: 200px; height: auto;">

# duxMLB - an extension

<p align="center">
  <img src="https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&color=blue" alt="Python" />
  <img src="https://img.shields.io/badge/Code-JavaScript-informational?style=flat&logo=javascript&color=yellow" alt="JavaScript" />
</p>

[![OS](https://img.shields.io/badge/OS-linux%2C%20windows%2C%20macOS-0078D4)](https://docs.abblix.com/docs/technical-requirements)

</div>

---

Developing an extension that can read your screen and give you live data updates and tips on your favourite baseball games! (With permission). 

> [!IMPORTANT]
> What can you do with duxMLB?

**duxMLB** offers two features
1. Panel page: For when you are watching a YouTube match and want to know about a certain player, or team or the schedule. (can be later extended to others as well)
2. Options page: For predicting your MLB future by comparing you to top players! If you have any legacy game in mind, upload it here for us to analyse and give you the statcast data!

There is also a `guide` page which explains how the site was made and contains all the techincal explanations. Visit this in the `extension options` and figure out what's in store for you!

## What is the panel page?

The panel page contains a `gemini` model that allows you to do the following

- [x] Ask the model about any players, team or MLB schedule and it will give `upto date` information (regards to APIs provided by Google)
- [x] You can, while watching a LIVE match, ask the model statcast data and it will analyize and answer!

> [!NOTE]
> What does that mean?

You can ask it questions like "Yoo! How fast was that pitch?" as soon as the pitcher finished pitching, and we will process the video in the backend, analyse the speed and tell it to you!

`duxMLB` is the most complete MLB guide that exists today.

---

# The workflow

![duxMLB](images/duxMLB_v3.png)

# The looks

This is the side panel. You can open this in any youtube video (currently works only for youtube).

![side panel](images/panel.png)

---

This is the options page. This page is to help you predict your MLB future and a how to guide on the **technical implementations** of the extension. It also has functionality to analyse **legacy MLB games** and generate statcast data for that.

![options page](images/options.png)

---

# Installing

I don't have a CWS developer account (cause I don't have 5 dollars), and hence I couldn't publish this on the webstore. So, I'll give the instructions to run this extension locally on your machine.

1. Clone the repository using

       git clone https://github.com/pUrGe12/MLBxG-extension

2. Now open chrome extensions using

       chrome://extensions

3. Toggle the `developer mode` button (on the top right) and then click on `load unpacked`

   ![Picture of developer mode button](./images/Screenshot1.png)

5. Select the `src` directory (inside the folder you cloned from github).

6. Click on `details` and visit the `options` page. This has more functionalities that cater to the other problems mentioned in the problem statment (and some fun stuff too).
    
   ![Picture of options](./images/Screenshot2.png)

8. Now you have the extension added.

Now you must create a `.env` file. You don't have to do that yourself, just run the python script and add the relevant values as and when it asks

        python3 createEnvFile.py

Then run the shell script to get selenium installed in your computer (even if you have google chrome, use "N" because you want to make sure it is compatible)

        chmod +x getPreReq.sh
        ./getPreReq.sh

Next step is to start the backend. There is just one main file that you need to run. From the root directory of the repo. Run the following commands to install the pre-requisities and run the backend

        pip install -r requirements.txt
        sudo apt install ffmpeg -y
        cd src/backend/ && python3 main.py



You can now start checking it out.

# How to test

1. Testing the panel

- Ask for a player's stats
- Ask for a team's stats
- Ask for the MLB schedule
- Ask for immediate statcast data while watching a live game
- Talk about any other baseball related things

2. Testing the options page

- Head over to the `options page` as mentioned above. Enter your MLB statcast to get a prediction
- Check out the guide for more details on how the project was implemented
- Upload a classics match (make it small otherwise you'll have to wait for long!) directly in MP4 format and generate the pitch speeds
- Type in the name for a match and generate the pitch speeds

---

# Work

- [x] Make the baseball detection thing run in parallel (THIS IS THE FIRST THING, START WITH THIS)

- [ ] In the second edition I am planning on annotating the video feed itself and displaying the entire video as an output. This will be much easier for the user to comprehend!
- [ ] Figure out the sending and recieving of the buffer video
- [ ] The novice users can enter a `mp4` video file that contains them practising their swings or pitches.
- [x] See if we can incorporate the curve of the ball during the pitch... (this we did using parabolic trajectory calculations)
- [ ] Write the readme for the math and the implementation of speed calculation precisely! Explain the use of scale factor precisely
- [ ] Create a schedule parser
- [ ] We'll probably have to switch from yolo to EfficientDet cause yolo is not google's.
How do we compare swings? Will have to learn...

# More things

- [ ] Integrating velocity finder with panel.py
- [ ] Creating a exit-velocity finder using bat-tracking
- [ ] Update logic to handle different camera angles. (to some extent, the math works --> Incorporating the scale factor)
- [x] Run statcast on the downloaded youtube video (at least the speed calculation works!)
