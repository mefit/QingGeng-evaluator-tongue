if [ "$1" = "dev" ]; then
  tmux \
    kill-session -t evaluator-tongue \; \
    new-session 'FLASK_ENV=development FLASK_APP=tongue pipenv run flask run' \; \
    split-window 'pipenv run celery -A tongue_tasks worker --loglevel=debug' \; \
    rename-session 'evaluator-tongue-devel' \;
else
  tmux \
    kill-session -t evaluator-tongue \; \
    new-session 'FLASK_APP=tongue pipenv run flask run' \; \
    split-window 'pipenv run celery -A tongue_tasks worker' \; \
    rename-session 'evaluator-tongue' \;
fi
