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

Options page

- [x] Finish the pinecone pipeline and prediction model
- [ ] Make it look better (add the similarity table in the right div)

---

Broad goals

- [ ] Figure out the sending and processing of buffer
- [ ] Train and test YOLOv5 on baseball detection in images
- [ ] Test if the pipeline works 
- [ ] Integrate the metric finding with panel.py

---

Now!

- [ ] Figure out the API setup today completely
- [ ] Write out the README for instructions regarding training of YOLOv5. 
- [ ] Start manually creating the dataset for baseballs.


# Installing

Create a virtual environment otherwise there might be version conflicts.

    python3 -m venv duxMLB
    source duxMLB/bin/activate

Then install the dependencies using

    pip install -r requirements.txt
