# MLBxG-extension

<p align="center">
  <img src="https://img.shields.io/badge/Code-Python-informational?style=flat&logo=python&color=blue" alt="Python" />
  <img src="https://img.shields.io/badge/Code-JavaScript-informational?style=flat&logo=javascript&color=yellow" alt="JavaScript" />
</p>

Developing an extension that can read your screen and give you live data updates and tips on your favourite baseball games! (With permission)

## The workflow

![duxMLB](images/duxMLB_V2.png)

## The looks

This is the side panel. You can open this in any youtube video (currently works only for youtube).

![side panel](images/panel.png)

---

This is the options page. This page is to help you predict your MLB future and a how to guide on the **technical implementations** of the extension. It involves the entire pipeline!

![options page](images/options.png)

---

Have created seperate ports for the panel and the options page because it will be much easier to manage and make changes in that.

## Work

Completed:

- [x] API setup and querying through the panel
- [x] Pinecone pipeline and prediction model
- [ ] Writing README for instructions regarding training YOLOv5
- [ ] Start manually creating the dataset for baseballs.

Options page

- [ ] Make it look better (add the similarity table in the right div)

---

Broad goals

- [ ] Train YOLOv5 on a better dataset
- [ ] Figure out the sending and recieving of the buffer video
- [ ] Integrate the finding the statcast data with panel.py

---

# Installing

Create a virtual environment otherwise there might be version conflicts.

    python3 -m venv duxMLB
    source duxMLB/bin/activate

Then install the dependencies using

    pip install -r requirements.txt
