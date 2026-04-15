#!/bin/bash
# Auto-annotate captured images for front, back, and no-card folders.
# Usage:
#   ./auto_annotate_all.sh front
#   ./auto_annotate_all.sh back
#   ./auto_annotate_all.sh none
#   ./auto_annotate_all.sh all

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

annotate() {
    local mode="$1"
    local class_name="$2"
    local input_dir="$PROJECT_ROOT/data/collected/$mode"

    if [ ! -d "$input_dir" ]; then
        echo "❌ Directory not found: $input_dir"
        echo "   Capture some images first using: python capture_server.py"
        return 1
    fi

    local count
    count=$(find "$input_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "⚠️  No images found in $input_dir — skipping"
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Annotating $mode → class: $class_name"
    echo "Images: $count"
    echo "=========================================="

    cd "$PROJECT_ROOT"
    source venv/bin/activate
    python training/scripts/auto_annotate.py -i "$input_dir" -c "$class_name"
}

case "${1:-all}" in
    front)
        annotate front id-front
        ;;
    back)
        annotate back id-back
        ;;
    none|no-card)
        annotate none no-card
        ;;
    all)
        annotate front id-front
        annotate back id-back
        annotate none no-card
        echo ""
        echo "✅ All annotations complete!"
        echo "Next: Open data/collected/*/ in LabelImg and refine bboxes."
        ;;
    *)
        echo "Usage: $0 [front|back|none|all]"
        echo ""
        echo "Examples:"
        echo "  $0 front     # Annotate front images only"
        echo "  $0 back      # Annotate back images only"
        echo "  $0 none      # Annotate no-card images only"
        echo "  $0 all       # Annotate all folders (default)"
        exit 1
        ;;
esac
