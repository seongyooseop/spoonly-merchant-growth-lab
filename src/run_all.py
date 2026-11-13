"""Rebuild everything from the simulated raw data: marts -> chapters 0-5 -> validation -> page and summary PDF."""
import build_marts
import analysis
import validate
import build_page

if __name__ == "__main__":
    build_marts.main()
    analysis.main()
    validate.main()
    build_page.main()
