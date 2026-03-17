import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app import create_app, db
from sqlalchemy import text, inspect

NEW_COLUMNS = [
    ("findings",         "TEXT"),
    ("condition",        "VARCHAR(200)"),
    ("severity",         "VARCHAR(60)"),
    ("recommendation",   "TEXT"),
    ("differential",     "TEXT"),
    ("explanation",      "TEXT"),
    ("validation_notes", "TEXT"),
    ("disclaimer",       "TEXT"),
]

def migrate():
    app = create_app()
    with app.app_context():
        inspector = inspect(db.engine)
        existing  = {c["name"] for c in inspector.get_columns("predictions")}
        added, skipped = [], []
        with db.engine.begin() as conn:
            for col, typ in NEW_COLUMNS:
                if col in existing:
                    skipped.append(col)
                    continue
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col} {typ}"))
                added.append(col)
                print(f"  + Added: predictions.{col}")
            if db.engine.dialect.name not in ("sqlite",):
                conn.execute(text("ALTER TABLE predictions ALTER COLUMN model_version TYPE VARCHAR(100)"))
        print("\n✅ Done.")
        if added:   print(f"   Added:   {', '.join(added)}")
        if skipped: print(f"   Skipped: {', '.join(skipped)}")

if __name__ == "__main__":
    migrate()
