#!/bin/bash

ADDNAME="markovplusm"

for I in 1 2 3 4; do
    for W in 48 96; do
        echo "=========================================="
        echo "Running configuration: idx=$I, window=$W"
        echo "=========================================="

        python chg_cmapsscfg.py --idx "$I" --window "$W" --addname "$ADDNAME"

        echo "Running main.py for cmapsst${ADDNAME}${I}_${W} ..."
        python main.py --name "cmapsst${ADDNAME}${I}_${W}" \
                       --config_file ./Config/cmapss.yaml \
                       --gpu 0 \
                       --sample 0 \
                       --milestone 150

        echo "Finished run for idx=$I, window=$W"
        echo
    done
done

echo "All runs completed."
