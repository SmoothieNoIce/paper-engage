.PHONY: run
run:
	python3 app.py

.PHONY: kill-port
kill-port:
	fuser -k 9200/tcp

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir=runs/ --port=6006
