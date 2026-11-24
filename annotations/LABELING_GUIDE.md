# Key-moment labelling guide

Unit: one 10-second window of transcript (`annotations/windows_<lecture>.tsv`).
Label `1` (key moment) when the window contains **any** of the following:

1. **Definition**: a new core concept is introduced or defined ("an algorithm is…", "a class is…").
2. **Principle or rule**: a general rule, property or takeaway that a student must remember
   (e.g. "integer division truncates", "big-O ignores constants").
3. **Explicit flag**: the lecturer says it is important, a key idea, a common mistake, or something to remember.
4. **Resolution**: the punchline or answer of a worked example (the result, why it works or fails).

Label `0` for: logistics and course admin, greetings, jokes and chit-chat, reading code line by line
without a new idea, transitions ("so let's move on"), repetition of something just said, silence,
and questions to the audience that the window does not answer.

Guidance:
- Judge only from the window's words. Use the neighbouring windows only to understand context.
- Aim for about 15–25 % positives. If more than 30 % look important, keep only the clearest.
- When a definition spans two windows, label the window where the defining sentence is mainly spoken.
