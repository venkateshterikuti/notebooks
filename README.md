# fm-pack-java-modernization

An Anteroom pack for repeatable, evidence-based Java modernization assessments.

Version `0.1.0` provides `/java-17-to-21`, a guidance-only workflow that inspects a local Java repository, identifies migration work, and validates proposed changes when the user explicitly approves command execution.

## Scope

Supported in this version:

- Java 17 to Java 21
- Maven and Gradle
- Gradle Groovy and Kotlin DSL
- Single-module and multi-module repositories
- Build, dependency, source, CI, container, and runtime configuration analysis
- Chat-only reports

Not supported in this version:

- Automatic source, build, or dependency modification
- Production deployment
- Organization-wide Java-version enforcement
- Full Spring Boot major-version migration
- Android, Scala-only, or Kotlin-only modernization
- A guarantee of production compatibility without application-specific testing

## Usage

Start Anteroom from the root of the target repository:

```powershell
cd C:\path\to\java-repository
aroom web
```

Run a read-only assessment:

```text
/java-17-to-21
```

Run the assessment and request post-change validation:

```text
/java-17-to-21 validate
```

The skill inspects files automatically. It must show the proposed commands and receive explicit approval before running Maven, Gradle, application, test, or bytecode-analysis commands.

## Output

The chat report includes:

- Overall readiness status
- Detected repository and Java configuration with file evidence
- Required changes
- Advisory recommendations
- Validation results
- Unresolved or high-risk items
- Checks not performed
- Suggested next steps

Possible statuses are `READY`, `CHANGES REQUIRED`, `BLOCKED`, `PARTIALLY VALIDATED`, and `UNSUPPORTED`.

## Install in the approved work environment

The onboarding script discovers this directory automatically when it is placed directly under `aroom-packs`.

Manual installation, when authorized in the work environment:

```powershell
aroom pack install <path-to-fm-pack-java-modernization>
aroom pack attach fm/java-modernization
aroom pack list
```

Do not run repository onboarding scripts outside the approved work environment.

## Baseline maintenance

The technical policy is in `instructions/java-17-to-21-baseline.md`. It is intentionally versioned and sourced so normal skill runs do not require internet access. Review the baseline periodically and before changing any exact compatibility recommendation.

## Validation evidence for the Jira story

In the work environment, capture:

1. Successful pack installation and attachment.
2. A Maven Java 17 assessment.
3. A Gradle Java 17 assessment.
4. Required versus advisory findings in each report.
5. A post-change validation report with remaining risks.
6. Java subject-matter-expert review of at least one report.

The `tests/fixtures` repositories provide small static-analysis examples. Run `python tests/validate_pack.py` only in an environment where Python execution is approved.
