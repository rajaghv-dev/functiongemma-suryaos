# Apps catalog — launchable applications for SuryaOS

Comprehensive list of applications the SuryaOS agent should recognize for the
`kde.krunner.launch` tool. Each app has multiple natural-language aliases
because users say "open the browser" more often than "launch firefox".

## Coverage

| Category | Count | Source |
|---|---|---|
| KDE core apps | 22 | Plasma 5.27 / 6.x default install |
| KDE utilities | 18 | Optional KDE applications |
| Open-source browsers | 18 | Linux-compatible, all FOSS or libre |
| Productivity / office | 12 | LibreOffice, Thunderbird, etc. |
| Media (audio/video/image) | 16 | VLC, GIMP, Audacity, Blender, etc. |
| Development tools | 14 | VS Code, JetBrains, Docker GUI, etc. |
| Communication | 10 | Signal, Element, Telegram, etc. |
| **Total** | **110** | apps with at least 3 aliases each |

Each app contributes:
- One canonical tool-call target (`kde.krunner.launch app=<binary>`)
- 3-7 natural-language phrasings as training examples
- Tokenizer entries for the binary name + common synonyms

## Files

| File | Purpose |
|---|---|
| `apps_catalog.json` | Canonical structured list (binary, aliases, category) |
| `launch_pairs.jsonl` | Training pairs in dispatch_pairs format |
| `app_aliases.txt` | Token list for tokenizer extension |

## Use in training

The `launch_pairs.jsonl` file follows the same format as `dispatch_pairs.jsonl`:
each line is one `(query, schema, tool_call)` example. To merge:

```bash
cat dataset/apps/launch_pairs.jsonl >> dataset/dispatch_pairs.jsonl
# Then dedupe by user query
```

For tokenizer extension, the script `training/build_tokenizer_dataset.py`
already pulls app names from `apps_catalog.json` automatically.

## Categories

### KDE core apps (always present on Plasma)
Dolphin, Kate, KWrite, Konsole, KMail, KOrganizer, KAddressBook,
KRunner, KCalc, KCharSelect, Spectacle, Gwenview, Okular, Ark,
Filelight, KSnapshot, KColorChooser, KFind, KFloppy, KSysGuard,
KHelpcenter, KInfoCenter

### KDE utilities (optional)
Krita, Kdenlive, KDevelop, KMag, KMenuEdit, KMix, KSystemLog,
KTorrent, KGet, Konqueror, Falkon, KStars, KGeography, Kanagram,
KTouch, Marble, Skanlite, Yakuake

### Open-source browsers
Firefox, Brave, Chromium, Ungoogled-Chromium, LibreWolf,
Tor Browser, Vivaldi, Falkon, GNOME Web (Epiphany), Min,
Qutebrowser, Pale Moon, Waterfox, Floorp, Iceweasel, IceCat,
Otter Browser, Midori

### Productivity / office
LibreOffice (Writer, Calc, Impress, Draw, Base, Math),
Thunderbird, Calligra Suite, OnlyOffice, FreeOffice,
Joplin, Logseq, Obsidian, Standard Notes, AbiWord,
Gnumeric, Zotero

### Media
VLC, MPV, Celluloid, SMPlayer, Haruna, Dragon Player,
Audacity, Ardour, MuseScore, LMMS, Hydrogen,
GIMP, Inkscape, Krita, Pinta, MyPaint,
Blender, Natron, OBS Studio, Kdenlive, OpenShot

### Development
VS Code, VSCodium, KDevelop, Qt Creator, GitHub Desktop,
GitKraken, Meld, KDiff3, Beyond Compare,
Postman, Insomnia, DBeaver, MySQL Workbench, pgAdmin

### Communication
Signal, Element (Matrix), Telegram, Discord, Slack,
Thunderbird, Trojita (KDE email), KMail, Riot, Fractal
