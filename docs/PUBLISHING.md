# Private repository upload and review

Run gh auth login --hostname github.com --git-protocol https --web in your own
terminal. Do not share the credential in chat or commit it.

After the candidate is audited and committed, from the repository root run:
python scripts/publish_private.py
For an organization use its existing GitHub owner name with --owner.
The helper refuses an existing repository name or an existing local origin,
creates only PRIVATE visibility, checks that property before push and verifies
the remote main commit. A network/auth failure is not an upload success.

The user authorized private creation/push, not public visibility. After a real
private URL exists, provide the file inventory, missing items, licensing and
security report for author review and explicit public approval. Confirm author
redistribution rights and original-code/data/weight/figure licenses before
making anything public.

Do not overwrite existing repositories. If creation or push fails after origin
has been set, inspect the recorded remote and actual private repository before
retrying; do not reset or replace them automatically.

Only after public approval and anonymous public-link verification should the
journal statements be finalized. An immutable release/archival DOI is optional
future work; none was created by this preparation.
