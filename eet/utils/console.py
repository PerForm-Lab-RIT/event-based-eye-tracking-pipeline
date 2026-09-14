
import io
import atexit


class Tee(io.TextIOBase):
  """A text stream that duplicates writes to multiple underlying streams.

  This is used to mirror stdout/stderr to a log file while keeping the
  original console output unchanged.
  """

  def __init__(self, *streams):
    super().__init__()
    # Filter out None and keep a stable order.
    self._streams = [s for s in streams if s is not None]

  def _alive_streams(self):
    alive = []
    for stream in self._streams:
      if getattr(stream, "closed", False):
        continue
      alive.append(stream)
    self._streams = alive
    return alive

  def write(self, s):
    # io.TextIOBase requires returning number of characters written.
    # Some streams may return None; we still return len(s).
    for stream in self._alive_streams():
      try:
        stream.write(s)
      except (ValueError, OSError):
        # Ignore closed/broken streams during interpreter shutdown.
        continue
    return len(s)

  def flush(self):
    for stream in self._alive_streams():
      try:
        stream.flush()
      except (ValueError, OSError):
        # Ignore closed/broken streams during interpreter shutdown.
        continue

  def isatty(self):
    # Preserve tty detection for progress bars etc.
    return any(hasattr(stream, "isatty") and stream.isatty() for stream in self._streams)


def setup_console_log(logdir, filename="console.log"):
  """Mirror stdout/stderr to a file under logdir.

  After calling this, anything written to stdout/stderr (print, tracebacks,
  etc.) will be visible both in the terminal and in the log file.

  Returns
  -------
  file handle
    The opened file handle so that the caller can manage its lifetime.
  """
  import sys

  # Line-buffered text file for timely flushing.
  path = logdir / filename
  f = path.open("a", buffering=1)
  orig_stdout = sys.stdout
  orig_stderr = sys.stderr
  tee_stdout = Tee(orig_stdout, f)
  tee_stderr = Tee(orig_stderr, f)
  sys.stdout = tee_stdout
  sys.stderr = tee_stderr

  def _cleanup_console_log():
    try:
      tee_stdout.flush()
      tee_stderr.flush()
    except Exception:
      pass

    if sys.stdout is tee_stdout:
      sys.stdout = orig_stdout
    if sys.stderr is tee_stderr:
      sys.stderr = orig_stderr

    try:
      if not f.closed:
        f.close()
    except Exception:
      pass

  atexit.register(_cleanup_console_log)
  return f
