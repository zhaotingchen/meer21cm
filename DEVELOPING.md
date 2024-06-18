## Installtion
In order to run the tests, you need to clone the repo and install the optional dependencies
```
pip install -e ".[full]"
```

A clean conda environment is recommended for this.

## Git workflow
You should always use Git for making changes to the code. To enforce a unified code style automatically, `pre-commit` is used. A typical workflow should look something like this:
- create a new branch based off main, either using a GUI for git in your IDE of choice or through CLI `git checkout -b new_branch_name` (make sure you are at main by running `git checkout main`)
- make your changes
- run the tests to make sure nothing breaks (see below).
- stage the changes (`git add files_you_changed`)
- run `git commit`. The first time will be `pre-commit` where tests are run. Files will be changed to conform to the *Black* code style.
- run `git commit -m "message you want to insert"` again.
- push to remote by running `git push -u origin new_branch_name`. If you have already pushed it before then ignore `-u`.
- Go to the github repo and create a pull request. **MAKE SURE** you ask for a review and have it approved with all tests passed. Branch protection rules may not be in place and in no circumstance should you merge into main without asking.
- After pull request is approved and merged, delete the branch on github.
- In your local repo, switch to main and then run `git pull`. Delete your local repo if you want.

## Tests
Whenever you have made some changes to the code, you want to make sure the code still passes all the tests, and preferably add new tests to cover the additions.

In the top level of the source directory, run:
```
pytest --cov=meerstack tests/
```

The above command run the tests and generate a report. Make sure there are no failed tests before you push.

You may want to add tests to cover more lines of code. There are multiple output formats available to check which lines are getting covered. I find html to be the easiest to read locally:
```
pytest --cov=meerstack tests/ --cov-report term --cov-report html:coverage.html
```
which will generate a folder `coverage.html` (do not commit it, leave it untracked). You can open the html files inside to see the coverage.
