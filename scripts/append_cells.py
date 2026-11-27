import json, os

nb_path = os.path.join('notebooks', 'run_C.ipynb')
with open(nb_path, encoding='utf-8') as f:
    nb = json.load(f)

cell6 = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "import csv\n",
        "for seed in (13, 42, 2026):\n",
        "    fp = f\"results/G10_M0+A0_dev_s{seed}.csv\"\n",
        "    try:\n",
        "        for r in csv.DictReader(open(fp, encoding=\"utf-8\")):\n",
        "            if r[\"correct\"] == \"0\" or r[\"ungrounded_cites\"] not in (\"\", \"0\"):\n",
        "                print(r[\"config\"], r[\"seed\"], r[\"session_id\"], r[\"turn\"], r[\"family\"],\n",
        "                      \"ungrounded=\" + r[\"ungrounded_cites\"],\n",
        "                      \"abst_exp=\" + r[\"abstain_expected\"], \"abst=\" + r[\"abstained\"],\n",
        "                      \"parse=\" + r[\"parse_ok\"], \"memrec=\" + r[\"memory_recall\"] )\n",
        "    except FileNotFoundError:\n",
        "        print(\"Missing:\", fp)\n"
    ],
    "outputs": [],
    "execution_count": None
}

cell7 = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "!python -m harness.summarize --group G10 --split dev\n",
        "!python -m harness.summarize --group G10 --split dev --latex > report/tables/table_2x2_dev.tex\n",
        "!python dashboard/make_dashboard.py --group G10 --splits dev test\n",
        "!python -m harness.trace_export --group G10\n"
    ],
    "outputs": [],
    "execution_count": None
}

nb.setdefault('cells', []).extend([cell6, cell7])

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Appended cells to', nb_path)
