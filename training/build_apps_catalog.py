#!/usr/bin/env python3
"""
Build the apps catalog and training pairs for app launching.

Produces:
    dataset/apps/apps_catalog.json    — structured app list
    dataset/apps/launch_pairs.jsonl   — training pairs (dispatch format)
    dataset/apps/app_aliases.txt      — flat list for tokenizer

Usage:
    python3 scripts/training/build_apps_catalog.py
    python3 scripts/training/build_apps_catalog.py --out ~/raja/functiongemma-suryaos/dataset/apps/

Each app generates 3-7 training examples covering its natural-language aliases:
    "open kate"                  → krunner_launch(app=kate)
    "launch the text editor"     → krunner_launch(app=kate)
    "start kate"                 → krunner_launch(app=kate)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# =============================================================================
# Apps catalog — (binary, category, aliases, description)
# =============================================================================
# Format: each entry is a dict with:
#   binary    — actual executable name passed to kstart/exec
#   category  — for grouping (kde_core / kde_util / browser / office / media / dev / comm)
#   aliases   — natural-language phrases users might say
#   desc      — short purpose for tokenizer corpus
# =============================================================================

APPS = [
    # ──────────────────────────────────────────────────────────────────────
    # KDE CORE — always present on Plasma desktop
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "dolphin",         "category": "kde_core",
     "aliases": ["dolphin", "file manager", "files", "file browser", "open files"],
     "desc": "KDE file manager"},
    {"binary": "kate",            "category": "kde_core",
     "aliases": ["kate", "text editor", "code editor", "editor"],
     "desc": "KDE advanced text editor"},
    {"binary": "kwrite",          "category": "kde_core",
     "aliases": ["kwrite", "simple text editor", "notepad"],
     "desc": "KDE simple text editor"},
    {"binary": "konsole",         "category": "kde_core",
     "aliases": ["konsole", "terminal", "shell", "command line", "cli"],
     "desc": "KDE terminal emulator"},
    {"binary": "kmail",           "category": "kde_core",
     "aliases": ["kmail", "email client", "mail", "email"],
     "desc": "KDE email client"},
    {"binary": "korganizer",      "category": "kde_core",
     "aliases": ["korganizer", "calendar", "scheduler"],
     "desc": "KDE calendar and todo"},
    {"binary": "kaddressbook",    "category": "kde_core",
     "aliases": ["kaddressbook", "address book", "contacts"],
     "desc": "KDE contact manager"},
    {"binary": "kcalc",           "category": "kde_core",
     "aliases": ["kcalc", "calculator", "calc"],
     "desc": "KDE calculator"},
    {"binary": "spectacle",       "category": "kde_core",
     "aliases": ["spectacle", "screenshot", "screen capture", "snip"],
     "desc": "KDE screenshot tool"},
    {"binary": "gwenview",        "category": "kde_core",
     "aliases": ["gwenview", "image viewer", "photo viewer", "picture viewer"],
     "desc": "KDE image viewer"},
    {"binary": "okular",          "category": "kde_core",
     "aliases": ["okular", "pdf viewer", "document viewer", "ebook reader"],
     "desc": "KDE document viewer"},
    {"binary": "ark",             "category": "kde_core",
     "aliases": ["ark", "archive manager", "zip extractor", "tar extractor"],
     "desc": "KDE archive manager"},
    {"binary": "filelight",       "category": "kde_core",
     "aliases": ["filelight", "disk usage analyzer", "disk space viewer"],
     "desc": "KDE disk usage visualizer"},
    {"binary": "krunner",         "category": "kde_core",
     "aliases": ["krunner", "run dialog", "command runner"],
     "desc": "KDE quick launcher"},
    {"binary": "systemsettings",  "category": "kde_core",
     "aliases": ["systemsettings", "settings", "system settings", "preferences", "control panel"],
     "desc": "KDE system settings"},
    {"binary": "kfind",           "category": "kde_core",
     "aliases": ["kfind", "file search", "find files"],
     "desc": "KDE file finder"},
    {"binary": "kcharselect",     "category": "kde_core",
     "aliases": ["kcharselect", "character selector", "special characters"],
     "desc": "KDE character map"},
    {"binary": "ksnapshot",       "category": "kde_core",
     "aliases": ["ksnapshot"],
     "desc": "KDE legacy screenshot tool"},
    {"binary": "ksysguard",       "category": "kde_core",
     "aliases": ["ksysguard", "system monitor", "task manager", "process viewer"],
     "desc": "KDE system monitor"},
    {"binary": "khelpcenter",     "category": "kde_core",
     "aliases": ["khelpcenter", "help", "documentation"],
     "desc": "KDE help center"},
    {"binary": "kinfocenter",     "category": "kde_core",
     "aliases": ["kinfocenter", "info center", "system info", "about my system"],
     "desc": "KDE hardware info viewer"},
    {"binary": "discover",        "category": "kde_core",
     "aliases": ["discover", "software center", "app store", "package manager gui"],
     "desc": "KDE software discovery"},

    # ──────────────────────────────────────────────────────────────────────
    # KDE UTILITIES — optional but common
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "krita",           "category": "kde_util",
     "aliases": ["krita", "digital paint", "drawing app"],
     "desc": "KDE digital painting"},
    {"binary": "kdenlive",        "category": "kde_util",
     "aliases": ["kdenlive", "video editor", "video editing"],
     "desc": "KDE non-linear video editor"},
    {"binary": "kdevelop",        "category": "kde_util",
     "aliases": ["kdevelop", "ide", "kde ide"],
     "desc": "KDE integrated development environment"},
    {"binary": "kmag",            "category": "kde_util",
     "aliases": ["kmag", "magnifier", "screen magnifier"],
     "desc": "KDE screen magnifier"},
    {"binary": "kmenuedit",       "category": "kde_util",
     "aliases": ["kmenuedit", "menu editor"],
     "desc": "KDE application menu editor"},
    {"binary": "kmix",            "category": "kde_util",
     "aliases": ["kmix", "mixer", "audio mixer", "volume mixer"],
     "desc": "KDE audio mixer"},
    {"binary": "ksystemlog",      "category": "kde_util",
     "aliases": ["ksystemlog", "system log", "log viewer"],
     "desc": "KDE log viewer"},
    {"binary": "ktorrent",        "category": "kde_util",
     "aliases": ["ktorrent", "torrent client", "torrent"],
     "desc": "KDE BitTorrent client"},
    {"binary": "kget",            "category": "kde_util",
     "aliases": ["kget", "download manager"],
     "desc": "KDE download manager"},
    {"binary": "konqueror",       "category": "kde_util",
     "aliases": ["konqueror"],
     "desc": "KDE web browser / file manager"},
    {"binary": "falkon",          "category": "kde_util",
     "aliases": ["falkon", "kde browser"],
     "desc": "KDE Qt-based web browser"},
    {"binary": "yakuake",         "category": "kde_util",
     "aliases": ["yakuake", "drop down terminal", "quake terminal"],
     "desc": "KDE drop-down terminal"},
    {"binary": "kstars",          "category": "kde_util",
     "aliases": ["kstars", "astronomy", "planetarium"],
     "desc": "KDE desktop planetarium"},
    {"binary": "marble",          "category": "kde_util",
     "aliases": ["marble", "globe", "world map"],
     "desc": "KDE virtual globe"},
    {"binary": "skanlite",        "category": "kde_util",
     "aliases": ["skanlite", "scanner", "scan"],
     "desc": "KDE scanner"},
    {"binary": "ksshaskpass",     "category": "kde_util",
     "aliases": ["ksshaskpass", "ssh key prompt"],
     "desc": "KDE SSH passphrase dialog"},
    {"binary": "kgpg",            "category": "kde_util",
     "aliases": ["kgpg", "encryption", "gpg gui"],
     "desc": "KDE GnuPG frontend"},
    {"binary": "partitionmanager", "category": "kde_util",
     "aliases": ["partitionmanager", "partition manager", "disk partitioner"],
     "desc": "KDE partition manager"},

    # ──────────────────────────────────────────────────────────────────────
    # OPEN-SOURCE BROWSERS — Firefox/Chromium families and minimal alternatives
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "firefox",         "category": "browser",
     "aliases": ["firefox", "browser", "web browser", "the internet", "ff"],
     "desc": "Mozilla Firefox web browser"},
    {"binary": "brave",           "category": "browser",
     "aliases": ["brave", "brave browser", "private browser"],
     "desc": "Brave privacy-focused browser"},
    {"binary": "chromium",        "category": "browser",
     "aliases": ["chromium", "chromium browser"],
     "desc": "Open-source Chromium browser"},
    {"binary": "google-chrome",   "category": "browser",
     "aliases": ["chrome", "google chrome"],
     "desc": "Google Chrome (proprietary, listed for completeness)"},
    {"binary": "ungoogled-chromium", "category": "browser",
     "aliases": ["ungoogled chromium", "ungoogled"],
     "desc": "Chromium with Google services stripped"},
    {"binary": "librewolf",       "category": "browser",
     "aliases": ["librewolf", "libre wolf"],
     "desc": "Privacy-focused Firefox fork"},
    {"binary": "torbrowser",      "category": "browser",
     "aliases": ["tor", "tor browser", "anonymous browser"],
     "desc": "Tor anonymity network browser"},
    {"binary": "vivaldi",         "category": "browser",
     "aliases": ["vivaldi"],
     "desc": "Vivaldi customizable browser"},
    {"binary": "epiphany",        "category": "browser",
     "aliases": ["epiphany", "gnome web", "gnome browser"],
     "desc": "GNOME web browser"},
    {"binary": "min",             "category": "browser",
     "aliases": ["min", "min browser", "minimal browser"],
     "desc": "Minimalist web browser"},
    {"binary": "qutebrowser",     "category": "browser",
     "aliases": ["qutebrowser", "qute", "keyboard browser", "vim browser"],
     "desc": "Keyboard-driven Vim-like browser"},
    {"binary": "palemoon",        "category": "browser",
     "aliases": ["pale moon", "palemoon"],
     "desc": "Pale Moon (Firefox fork)"},
    {"binary": "waterfox",        "category": "browser",
     "aliases": ["waterfox"],
     "desc": "Waterfox (Firefox fork)"},
    {"binary": "floorp",          "category": "browser",
     "aliases": ["floorp"],
     "desc": "Floorp (Firefox fork)"},
    {"binary": "icecat",          "category": "browser",
     "aliases": ["icecat", "gnu icecat"],
     "desc": "GNU IceCat (Firefox fork)"},
    {"binary": "otter-browser",   "category": "browser",
     "aliases": ["otter", "otter browser"],
     "desc": "Otter classic Opera-like browser"},
    {"binary": "midori",          "category": "browser",
     "aliases": ["midori"],
     "desc": "Midori lightweight browser"},
    {"binary": "nyxt",            "category": "browser",
     "aliases": ["nyxt", "lisp browser"],
     "desc": "Nyxt power-user browser"},

    # ──────────────────────────────────────────────────────────────────────
    # PRODUCTIVITY / OFFICE
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "libreoffice",     "category": "office",
     "aliases": ["libreoffice", "office", "office suite"],
     "desc": "LibreOffice main launcher"},
    {"binary": "libreoffice --writer", "category": "office",
     "aliases": ["libreoffice writer", "writer", "word processor", "document editor"],
     "desc": "LibreOffice Writer"},
    {"binary": "libreoffice --calc",   "category": "office",
     "aliases": ["libreoffice calc", "calc", "spreadsheet", "excel"],
     "desc": "LibreOffice Calc spreadsheet"},
    {"binary": "libreoffice --impress", "category": "office",
     "aliases": ["libreoffice impress", "impress", "presentation", "slideshow"],
     "desc": "LibreOffice Impress presentation"},
    {"binary": "thunderbird",     "category": "office",
     "aliases": ["thunderbird", "tbird"],
     "desc": "Mozilla Thunderbird email"},
    {"binary": "joplin",          "category": "office",
     "aliases": ["joplin", "notes", "note taking"],
     "desc": "Joplin note-taking"},
    {"binary": "logseq",          "category": "office",
     "aliases": ["logseq"],
     "desc": "Logseq knowledge base"},
    {"binary": "obsidian",        "category": "office",
     "aliases": ["obsidian"],
     "desc": "Obsidian markdown notes"},
    {"binary": "zotero",          "category": "office",
     "aliases": ["zotero", "reference manager", "citations"],
     "desc": "Zotero research assistant"},
    {"binary": "calligra",        "category": "office",
     "aliases": ["calligra", "calligra suite"],
     "desc": "KDE Calligra office suite"},
    {"binary": "onlyoffice-desktopeditors", "category": "office",
     "aliases": ["onlyoffice"],
     "desc": "OnlyOffice editors"},
    {"binary": "abiword",         "category": "office",
     "aliases": ["abiword"],
     "desc": "AbiWord lightweight word processor"},

    # ──────────────────────────────────────────────────────────────────────
    # MEDIA — audio, video, image
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "vlc",             "category": "media",
     "aliases": ["vlc", "video player", "media player"],
     "desc": "VLC media player"},
    {"binary": "mpv",             "category": "media",
     "aliases": ["mpv"],
     "desc": "mpv media player"},
    {"binary": "celluloid",       "category": "media",
     "aliases": ["celluloid"],
     "desc": "Celluloid mpv frontend"},
    {"binary": "haruna",          "category": "media",
     "aliases": ["haruna", "kde video player"],
     "desc": "Haruna KDE video player"},
    {"binary": "audacity",        "category": "media",
     "aliases": ["audacity", "audio editor", "sound editor"],
     "desc": "Audacity audio editor"},
    {"binary": "ardour",          "category": "media",
     "aliases": ["ardour", "daw", "audio workstation"],
     "desc": "Ardour digital audio workstation"},
    {"binary": "musescore",       "category": "media",
     "aliases": ["musescore", "music notation", "sheet music"],
     "desc": "MuseScore music notation"},
    {"binary": "lmms",            "category": "media",
     "aliases": ["lmms"],
     "desc": "LMMS music production"},
    {"binary": "gimp",            "category": "media",
     "aliases": ["gimp", "image editor", "photo editor", "photoshop alternative"],
     "desc": "GIMP image editor"},
    {"binary": "inkscape",        "category": "media",
     "aliases": ["inkscape", "vector editor", "svg editor", "illustrator alternative"],
     "desc": "Inkscape vector graphics"},
    {"binary": "blender",         "category": "media",
     "aliases": ["blender", "3d", "3d modeling", "3d editor"],
     "desc": "Blender 3D creation suite"},
    {"binary": "obs",             "category": "media",
     "aliases": ["obs", "obs studio", "screen recorder", "stream recorder"],
     "desc": "OBS Studio recording/streaming"},
    {"binary": "openshot",        "category": "media",
     "aliases": ["openshot"],
     "desc": "OpenShot video editor"},
    {"binary": "shotcut",         "category": "media",
     "aliases": ["shotcut"],
     "desc": "Shotcut video editor"},
    {"binary": "darktable",       "category": "media",
     "aliases": ["darktable", "raw editor", "photo lab"],
     "desc": "darktable photo workflow"},
    {"binary": "rawtherapee",     "category": "media",
     "aliases": ["rawtherapee", "raw photo editor"],
     "desc": "RawTherapee RAW processor"},

    # ──────────────────────────────────────────────────────────────────────
    # DEVELOPMENT
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "code",            "category": "dev",
     "aliases": ["vscode", "vs code", "visual studio code", "code"],
     "desc": "Visual Studio Code editor"},
    {"binary": "codium",          "category": "dev",
     "aliases": ["vscodium", "codium"],
     "desc": "VSCodium (open VS Code)"},
    {"binary": "qtcreator",       "category": "dev",
     "aliases": ["qtcreator", "qt creator", "qt ide"],
     "desc": "Qt Creator IDE"},
    {"binary": "github-desktop",  "category": "dev",
     "aliases": ["github desktop"],
     "desc": "GitHub Desktop"},
    {"binary": "gitkraken",       "category": "dev",
     "aliases": ["gitkraken", "git client"],
     "desc": "GitKraken git GUI"},
    {"binary": "gitg",            "category": "dev",
     "aliases": ["gitg", "gnome git"],
     "desc": "gitg GNOME git GUI"},
    {"binary": "meld",            "category": "dev",
     "aliases": ["meld", "diff tool", "diff viewer"],
     "desc": "Meld diff/merge tool"},
    {"binary": "kdiff3",          "category": "dev",
     "aliases": ["kdiff3", "kdiff"],
     "desc": "KDiff3 three-way merge"},
    {"binary": "postman",         "category": "dev",
     "aliases": ["postman", "api tester"],
     "desc": "Postman API tool"},
    {"binary": "insomnia",        "category": "dev",
     "aliases": ["insomnia", "rest client"],
     "desc": "Insomnia API client"},
    {"binary": "dbeaver",         "category": "dev",
     "aliases": ["dbeaver", "database tool", "sql client"],
     "desc": "DBeaver SQL client"},
    {"binary": "mysql-workbench", "category": "dev",
     "aliases": ["mysql workbench", "mysql gui"],
     "desc": "MySQL Workbench"},
    {"binary": "pgadmin4",        "category": "dev",
     "aliases": ["pgadmin", "postgresql gui"],
     "desc": "pgAdmin Postgres GUI"},
    {"binary": "android-studio",  "category": "dev",
     "aliases": ["android studio"],
     "desc": "Android Studio IDE"},

    # ──────────────────────────────────────────────────────────────────────
    # COMMUNICATION
    # ──────────────────────────────────────────────────────────────────────
    {"binary": "signal-desktop",  "category": "comm",
     "aliases": ["signal", "signal messenger"],
     "desc": "Signal secure messenger"},
    {"binary": "element-desktop", "category": "comm",
     "aliases": ["element", "matrix", "matrix client"],
     "desc": "Element Matrix client"},
    {"binary": "telegram-desktop", "category": "comm",
     "aliases": ["telegram", "telegram desktop"],
     "desc": "Telegram messenger"},
    {"binary": "discord",         "category": "comm",
     "aliases": ["discord"],
     "desc": "Discord chat"},
    {"binary": "slack",           "category": "comm",
     "aliases": ["slack"],
     "desc": "Slack workspace messaging"},
    {"binary": "fractal",         "category": "comm",
     "aliases": ["fractal", "gtk matrix client"],
     "desc": "Fractal Matrix client"},
    {"binary": "dino",            "category": "comm",
     "aliases": ["dino", "xmpp", "jabber"],
     "desc": "Dino XMPP client"},
    {"binary": "trojita",         "category": "comm",
     "aliases": ["trojita", "qt email"],
     "desc": "Trojitá Qt email client"},
    {"binary": "bitwarden",       "category": "comm",
     "aliases": ["bitwarden", "password manager"],
     "desc": "Bitwarden password manager"},
    {"binary": "keepassxc",       "category": "comm",
     "aliases": ["keepassxc", "keepass", "offline password manager"],
     "desc": "KeePassXC password manager"},
]


# =============================================================================
# Training pair generation
# =============================================================================

LAUNCH_PHRASINGS = [
    "open {alias}",
    "launch {alias}",
    "start {alias}",
    "run {alias}",
    "open up {alias}",
]


def make_pair(query: str, app_binary: str, schema: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Call the right tool."},
            {"role": "user",   "content": query},
        ],
        "tools": [schema],
        "target": {"name": "kde_krunner_launch", "arguments": {"app": app_binary}},
        "source": "apps_catalog",
    }


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path.home() / "raja/functiongemma-suryaos/dataset/apps",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the krunner_launch schema for every training example.
    sys.path.insert(0, str(ROOT / "src"))
    from surya_agent.tool_registry import ToolRegistry
    from surya_agent.context_builder import _tool_to_mcp_schema
    reg = ToolRegistry.load(ROOT / "tools")
    schema = _tool_to_mcp_schema(reg.tools["kde.krunner.launch"])

    # ------- 1. Catalog JSON -------
    print(f"[1/3] Writing catalog ({len(APPS)} apps)...")
    catalog_path = out_dir / "apps_catalog.json"
    by_cat: dict[str, list[dict]] = {}
    for app in APPS:
        by_cat.setdefault(app["category"], []).append(app)
    catalog_path.write_text(json.dumps({
        "total":   len(APPS),
        "categories": {cat: len(items) for cat, items in by_cat.items()},
        "apps":   APPS,
    }, indent=2))
    print(f"    ✓ {catalog_path}")
    for cat, items in sorted(by_cat.items()):
        print(f"      {cat:12s} {len(items):3d}")

    # ------- 2. Training pairs -------
    print(f"[2/3] Generating launch_pairs.jsonl...")
    pairs: list[dict] = []
    for app in APPS:
        for alias in app["aliases"]:
            for template in LAUNCH_PHRASINGS:
                query = template.format(alias=alias)
                pairs.append(make_pair(query, app["binary"], schema))

    pairs_path = out_dir / "launch_pairs.jsonl"
    with open(pairs_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"    ✓ {pairs_path}  ({len(pairs)} pairs)")

    # ------- 3. Aliases for tokenizer -------
    print(f"[3/3] Writing app_aliases.txt for tokenizer...")
    aliases_path = out_dir / "app_aliases.txt"
    all_aliases: list[str] = []
    for app in APPS:
        all_aliases.append(app["binary"])
        all_aliases.extend(a for a in app["aliases"] if " " not in a and len(a) > 2)
    # Dedupe, preserve order
    seen, dedup = set(), []
    for a in all_aliases:
        if a.lower() not in seen:
            seen.add(a.lower())
            dedup.append(a)
    aliases_path.write_text("\n".join(dedup) + "\n")
    print(f"    ✓ {aliases_path}  ({len(dedup)} unique aliases)")

    print()
    print("Done.")
    print(f"  Apps:        {len(APPS)}")
    print(f"  Categories:  {len(by_cat)}")
    print(f"  Train pairs: {len(pairs)}")
    print(f"  Aliases:     {len(dedup)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
