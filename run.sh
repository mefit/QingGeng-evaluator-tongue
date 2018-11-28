tmux \
  has-session -t evaluator-tongue \; \
  kill-session -t evaluator-tongue \;

if [ "$1" = "dev" ]; then
  tmux \
    new-session 'FLASK_ENV=development FLASK_APP=tongue pipenv run flask run' \; \
    split-window 'pipenv run celery -A tongue_tasks worker --loglevel=debug' \; \
    rename-session 'evaluator-tongue-devel' \;
else
  tmux \
    new-session 'FLASK_APP=tongue pipenv run flask run' \; \
    split-window 'pipenv run celery -A tongue_tasks worker' \; \
    rename-session 'evaluator-tongue' \;
fi
