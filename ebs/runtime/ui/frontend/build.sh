#!/bin/bash

npm run build
# cp -r ./src/assets/* ./dist/assets # workaround
cp -r ./dist ./ebs_agent_ui/static

python -m build --outdir build && echo "build success"

echo "clean up static"
rm -r ./ebs_agent_ui/static