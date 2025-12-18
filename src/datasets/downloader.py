"""
Dataset download utilities.
"""

import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download_file(
    url: str,
    dest_path: str,
    chunk_size: int = 8192,
    show_progress: bool = True,
    checksum_md5: Optional[str] = None,
) -> str:
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination file path
        chunk_size: Download chunk size
        show_progress: Show progress bar
        checksum_md5: Expected MD5 checksum

    Returns:
        Path to downloaded file
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already exists with correct checksum
    if dest_path.exists() and checksum_md5:
        if verify_checksum(str(dest_path), checksum_md5):
            print(f"File already exists with correct checksum: {dest_path}")
            return str(dest_path)

    # Download
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    if show_progress:
        progress = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=dest_path.name,
        )

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                if show_progress:
                    progress.update(len(chunk))

    if show_progress:
        progress.close()

    # Verify checksum
    if checksum_md5:
        if not verify_checksum(str(dest_path), checksum_md5):
            raise ValueError(f"Checksum mismatch for {dest_path}")

    return str(dest_path)


def download_ftp(
    url: str,
    dest_path: str,
    show_progress: bool = True,
) -> str:
    """
    Download a file from FTP URL.

    Args:
        url: FTP URL
        dest_path: Destination path

    Returns:
        Path to downloaded file
    """
    import ftplib

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    host = parsed.netloc
    filepath = parsed.path

    ftp = ftplib.FTP(host)
    ftp.login()

    # Get file size
    ftp.voidcmd('TYPE I')
    size = ftp.size(filepath)

    if show_progress:
        progress = tqdm(
            total=size,
            unit='iB',
            unit_scale=True,
            desc=dest_path.name,
        )

    with open(dest_path, 'wb') as f:
        def callback(data):
            f.write(data)
            if show_progress:
                progress.update(len(data))

        ftp.retrbinary(f'RETR {filepath}', callback)

    if show_progress:
        progress.close()

    ftp.quit()
    return str(dest_path)


def extract_archive(
    archive_path: str,
    extract_dir: str,
    remove_archive: bool = False,
) -> str:
    """
    Extract a tar.gz or zip archive.

    Args:
        archive_path: Path to archive
        extract_dir: Directory to extract to
        remove_archive: Delete archive after extraction

    Returns:
        Path to extraction directory
    """
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {archive_path.name}...")

    if archive_path.suffix == '.gz' or str(archive_path).endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
    elif archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")

    if remove_archive:
        archive_path.unlink()

    return str(extract_dir)


def verify_checksum(filepath: str, expected_md5: str) -> bool:
    """
    Verify MD5 checksum of a file.

    Args:
        filepath: Path to file
        expected_md5: Expected MD5 hash

    Returns:
        True if checksum matches
    """
    md5_hash = hashlib.md5()

    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)

    return md5_hash.hexdigest() == expected_md5


def get_cache_dir() -> Path:
    """Get the cache directory for downloads."""
    cache_dir = Path(os.environ.get('VECTORDB_CACHE', Path.home() / '.cache' / 'vectordb_bench'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
