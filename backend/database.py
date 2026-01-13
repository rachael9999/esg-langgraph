from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import func
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Example: mysql+pymysql://user:password@localhost/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db") 

try:
    engine = create_engine(DATABASE_URL, pool_recycle=3600)
    # Try to connect to verify credentials (eager check)
    with engine.connect() as conn:
        pass
except Exception as e:
    print(f"CRITICAL WARNING: Database connection to {DATABASE_URL} failed: {e}")
    print("Falling back to SQLite (local file) temporarily.")
    DATABASE_URL = "sqlite:///./fallback_esg.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBSession(Base):
    __tablename__ = "sessions"

    session_id = Column(String(36), primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    documents = relationship("DBDocument", back_populates="session")

class DBDocument(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(36), ForeignKey("sessions.session_id"))
    filename = Column(String(255), nullable=False)
    page_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    summary = Column(Text, nullable=True)
    vector_ids = Column(Text, nullable=True)  # JSON string of list of IDs

    session = relationship("DBSession", back_populates="documents")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
