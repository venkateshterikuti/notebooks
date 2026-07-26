# README

# fm-pack-java-modernization

An Anteroom pack for durable, evidence-based Java 17-to-21 assessment,
approval-gated migration, validation, bounded repair, and reporting.

Version `0.4.0` adds the `fm_java_17_to_21` workflow template. The original
`/java-17-to-21` skill remains as a read-only interactive facade; durable
source changes now run through persisted workflow phases and human gates.

## Workflow

The workflow executes:

1. static assessment and complete release-delta coverage
2. selected-scope exact proposal persisted as a dossier and unified diff
3. top-level human approval of that exact initial diff
4. application of only the approved patch
5. permissioned Java/build/test validation
6. up to two explicit repair proposal, approval, application, and revalidation
   rounds
7. final readiness, change, validation, consequence, and rollback report

The repair rounds are intentionally explicit rather than a native nested loop.
This keeps every human gate top-level and durable.

Workflow state is written at the target repository root using the default
prefix `.anteroom-java-21`:

- `.anteroom-java-21-assessment.md`
- `.anteroom-java-21-proposal.md`
- `.anteroom-java-21-proposal.patch`
- `.anteroom-java-21-validation-<round>.md`
- `.anteroom-java-21-repair-<round>.md`
- `.anteroom-java-21-repair-<round>.patch`

State-artifact writes are not source-edit approval. The workflow still requires
a human decision for every exact source or repair patch. Edit approval does not
authorize Maven, Gradle, wrapper, test, application, `jdeps`, or `jdeprscan`
commands. Those commands use Anteroom's separate `Allow Once`,
`Allow for Session`, or rejection flow.

## Scope parameters

Pass workflow parameters with repeated `--param KEY=VALUE` arguments:

- `repository_path`: target repository, default `.`
- `change_scope`: `required`, `safe`, `modernize`, or `all`; default `required`
- `validation_profile`: `standard` or `none`; default `standard`
- `state_prefix`: workflow artifact prefix; default `.anteroom-java-21`

`all` includes Design Sensitive and Experimental proposals only when their
warnings, failure consequences, portability impact, rollback, and validation
are present in the persisted dossier.

## Install and run

From the directory containing this pack:

```powershell
aroom pack install .\fm-pack-java-modernization
```

From the target Java repository:

```powershell
aroom pack attach fm/java-modernization --project
aroom workflow run fm_java_17_to_21 --dry-run
aroom workflow run fm_java_17_to_21
```

Example with explicit parameters:

```powershell
aroom workflow run fm_java_17_to_21 `
  --param change_scope=safe `
  --param validation_profile=standard
```

When a run pauses, inspect it before deciding:

```powershell
aroom workflow status <RUN_ID>
aroom workflow history <RUN_ID>
aroom workflow transcript <RUN_ID>
```

Resolve a source-diff human gate:

```powershell
aroom workflow respond <RUN_ID> --option approve_diff
```

Resolve a repair gate:

```powershell
aroom workflow respond <RUN_ID> --option approve_repair
```

Approve or deny a pending tool execution request:

```powershell
aroom workflow approve <RUN_ID>
aroom workflow deny <RUN_ID>
```

Continue or diagnose as needed:

```powershell
aroom workflow resume <RUN_ID>
aroom workflow why <RUN_ID>
aroom workflow diagnose <RUN_ID>
aroom workflow replay <RUN_ID>
```

## Safety boundaries

- No source edit occurs before approval of the exact persisted patch.
- A stale or mismatched patch stops without broadening the edit.
- A failed validation never authorizes an automatic repair.
- Every repair has a new exact patch and a new human gate.
- The workflow never commits, pushes, deploys, installs tools, or modifies
  external systems.
- Optional modernization remains separate from Java 21 readiness.
- `READY` requires successful runtime validation and no required or high-risk
  unresolved findings.

## Interactive facade

Use `/java-17-to-21`, `/java-17-to-21 audit`, or
`/java-17-to-21 validate` for one-off read-only work. `/java-17-to-21 apply`
and the legacy `plan` mode direct users to `fm_java_17_to_21`.

## Pack validation

The dependency-free structural test remains:

```powershell
python .\tests\validate_pack.py
```

Anteroom performs the authoritative workflow-schema checks:

```powershell
aroom workflow validate .\workflow_templates\fm-java-17-to-21.yaml
aroom workflow run .\workflow_templates\fm-java-17-to-21.yaml --dry-run
```

Do not use `aroom workflow simulate` as an acceptance gate for this template
unless the company build wires installed skills into the simulator.
Skill-backed runners can otherwise report a false missing `SkillRegistry`
failure even though validation and normal execution are correctly configured.



# Baseline

# Java 17 to Java 21 Migration Baseline

**Baseline version:** 0.4.0  
**Last reviewed:** 2026-07-26  
**Status:** Initial modernization-enabled baseline pending organizational Java SME approval

## Purpose

Use this baseline with the `/java-17-to-21` skill to produce consistent, evidence-based migration guidance and approval-gated implementation and validation. Treat it as guidance, not proof that an application is production-ready.

This baseline governs Java 21 migration readiness. The `java-17-to-21-release-delta` instruction defines the complete JEP-level coverage boundary. Optional modernization work is governed by the `java-17-to-21-modernization-policy` and `java-17-to-21-pattern-catalog` instructions. Readiness statuses remain separate from modernization progress.

## Mandatory behavior

- Inspect only the current local workspace unless the user explicitly identifies another local path.
- Do not install a JDK, build tool, or scanner. Dependency/plugin edits require exact-diff approval, and any resulting downloads require command execution permission.
- Do not push, commit, open a pull request, deploy, or update Jira/Confluence.
- Do not send repository content to online services merely to perform the assessment.
- Redact secrets and sensitive values encountered in configuration.
- State which checks were performed, skipped, failed, or could not be verified.
- Never report `READY` when required runtime validation was skipped or failed.
- Never treat optional modernization opportunities as Java 21 readiness blockers without concrete evidence.
- Never apply source changes without explicit user approval of the exact implementation set.
- Treat suggestion selection, edit approval, and repository-code execution approval as separate decisions. Approval of edits never authorizes build, test, application, or analysis commands.

## Supported scope

Support Java source repositories that declare or credibly target Java 17 and use:

- Maven, including Maven Wrapper and multi-module projects
- Gradle, including Gradle Wrapper, Groovy DSL, Kotlin DSL, and multi-project builds
- Common CI, container, buildpack, and Java-version configuration
- Java source modernization audit for language, API, concurrency, and delivery/runtime cleanup opportunities

Framework migrations such as Spring Boot may be proposed and implemented by the durable workflow after Design Sensitive warnings, exact-diff approval, and framework-specific validation. Treat Android, Scala-only, Kotlin-only, Ant-only, Bazel-only, and other build systems as `UNSUPPORTED` in version 0.4.0. For mixed-language projects, assess Java/build configuration and mark non-Java compilation compatibility `UNRESOLVED`.

## Evidence and classification policy

For every finding, include a file path and the relevant property, plugin, dependency, source pattern, or command result. Also record the related release/JEP when applicable, evidence type (`STATIC` or `VALIDATED`), confidence (`HIGH`, `MEDIUM`, or `LOW`), and applicability. If an effective value is inherited from a remote parent, convention plugin, environment, or unavailable CI variable, say so.

Classify findings as:

- **REQUIRED:** Concrete repository or command evidence shows a change is necessary to build, test, or run correctly on Java 21, or a required Java 21 setting remains on Java 17.
- **ADVISORY:** A resilience, maintainability, or future-readiness improvement that is not proven to block Java 21.
- **UNRESOLVED:** Compatibility cannot be established from available evidence or an authoritative support statement is missing.
- **MODERNIZATION_CANDIDATE:** A source-level opportunity to adopt newer Java language, API, or concurrency features without claiming it is required for migration readiness.
- **EXPERIMENTAL:** A preview, incubating, or design-sensitive opportunity that requires explicit opt-in and separate validation and must never be treated as a required migration step.

Do not label a dependency incompatible solely because it is old. Exact version recommendations require a dated, authoritative vendor/project source or an approved organizational standard. Otherwise recommend upgrading to an organization-approved Java 21-compatible version and classify compatibility as unresolved.

## Configuration discovery baseline

Inspect, when present:

- `pom.xml`, parent POM references, modules, profiles, properties, dependency management, plugin management, and `.mvn/`
- `build.gradle`, `build.gradle.kts`, `settings.gradle`, `settings.gradle.kts`, `gradle.properties`, version catalogs, convention plugins, included builds, and `gradle/wrapper/gradle-wrapper.properties`
- `mvnw`, `mvnw.cmd`, `gradlew`, and `gradlew.bat`
- `.java-version`, `.sdkmanrc`, `.tool-versions`, `jenv` files, Maven toolchain declarations, Gradle toolchains, and IDE compiler settings when committed
- Dockerfiles, buildpacks, deployment manifests, startup scripts, service files, and JVM option files
- Bitbucket Pipelines, Jenkinsfiles, and other CI configuration
- All modules, not just the repository root

Report contradictions separately. For example, a build targeting 21 while CI or the runtime image still uses 17 is a required alignment finding.

## Maven baseline

- Prefer Maven Wrapper when present; record the wrapper distribution version.
- Detect `maven.compiler.release`, compiler-plugin `<release>`, `<source>`, `<target>`, `java.version`, profile overrides, and inherited values.
- Prefer `--release` or `maven.compiler.release` over separate source and target values because it validates API availability for the selected Java release.
- Treat a remaining effective release/source/target of 17 as `REQUIRED` for a migration whose intended output is Java 21 bytecode.
- Do not invent a universal minimum Maven version. Verify Maven core, wrapper, compiler, Surefire, Failsafe, Javadoc, JaCoCo, shading, packaging, and framework plugin compatibility using authoritative documentation or an approved baseline.
- Treat Maven Toolchains as `ADVISORY` unless organization policy or the repository requires reproducible JDK selection; a misconfigured required toolchain is `REQUIRED`.
- Inspect parent POMs and plugin management before concluding which versions are effective.

## Gradle baseline

- Prefer Gradle Wrapper when present; extract the distribution version from `gradle-wrapper.properties`.
- Detect Java toolchains, `sourceCompatibility`, `targetCompatibility`, `options.release`, test launcher configuration, convention plugins, and per-project overrides.
- Java 21 toolchain support begins with Gradle 8.4; running Gradle itself on Java 21 is supported by Gradle 8.5 and later. A wrapper older than the applicable support level is `REQUIRED` when the build will compile/test with or run on Java 21.
- Do not recommend an uncontrolled jump to the newest Gradle major. Recommend a supported, organization-approved version and note intermediate upgrade guides when relevant.
- Inspect build logic, included builds, `buildSrc`, plugin versions, Kotlin/Groovy versions, and version catalogs before concluding compatibility.
- Prefer Java toolchains for reproducible compiler/test launcher selection; classify as `ADVISORY` unless policy or repository behavior makes it required.

## Dependency and plugin baseline

Inventory direct versions, managed versions, BOMs, parent-managed versions, Gradle platforms, version catalogs, annotation processors, agents, and build plugins. Prioritize compatibility review for:

- Application frameworks and embedded servers
- Lombok and other annotation processors
- Byte Buddy, ASM, cglib, Mockito, and reflection/bytecode libraries
- JaCoCo and other coverage/instrumentation tools
- APM, profiling, security, and runtime Java agents
- Serialization, XML/binding, scripting, JNI/JNA, database drivers, and native libraries
- Maven/Gradle plugins that execute inside the build JVM

Framework major-version migration is implementable when the skill can produce a concrete repository-local diff, disclose Design Sensitive consequences and known unknowns, obtain exact approval, and run the framework-specific validation the user permits. If evidence is insufficient to prepare a responsible diff, mark the implementation `BLOCKED` rather than claiming the framework is compatible.

## Source and runtime risk baseline

Inspect for evidence of:

- Internal JDK APIs (`sun.*`, most `com.sun.*`, `jdk.internal.*`) and `Unsafe`
- Reflective access and `--add-opens`, `--add-exports`, or `--illegal-access`
- APIs deprecated for removal or removed between JDK 17 and JDK 21
- Custom `finalize()` methods or finalization dependence
- Dynamic agent attachment and warnings introduced by JDK 21
- Implicit default-charset use, especially file and stream readers/writers, scanners, formatters, source encoding, and Windows-originated data
- Obsolete or removed JVM or GC flags
- Security providers, TLS or crypto behavior, certificate stores, locale-sensitive behavior, and native integrations that require environment-specific testing

Static source matches are indicators, not proof. Confirm with compilation, tests, `jdeps`, `jdeprscan`, runtime logs, or authoritative library documentation when available.

An `--add-opens` flag indicates reflective access to a strongly encapsulated package. It does not, by itself, prove use of an internal JDK API. Report the package opened, identify the library or code path that requires it when possible, and require Java 21 test evidence before recommending removal.

## Java 17 to 21 changes emphasized by this baseline

- JDK 18 and later use UTF-8 as the default charset. Applications that implicitly depended on an environment-specific JDK 17 charset require behavioral testing and may require explicit charsets.
- Finalization was deprecated for removal in JDK 18. Custom finalizers and libraries relying on finalization require review; migration to explicit resource management is advisory unless current behavior fails or policy requires removal.
- JDK 21 warns when agents are loaded dynamically. Identify test, instrumentation, and APM behavior and distinguish dynamic attachment from agents loaded at JVM startup.
- JDK releases may remove or deprecate tools, APIs, security algorithms, and JVM options. Use the JDK 21 migration guide and release notes rather than relying on memory.

## Release-delta coverage baseline

Walk every entry in `java-17-to-21-release-delta`. Record `APPLICABLE_FINDING`, `APPLICABLE_NO_FINDING`, `NOT_APPLICABLE`, `UNRESOLVED`, or `NOT_CHECKED` for each entry. This coverage table is required in `audit` and `apply` modes and may be summarized in assessment/validate reports when the detailed table would obscure readiness findings.

The completeness claim is JEP-level plus the Oracle migration guide's significant changes and removed/deprecated API checks. It is not a claim that every JDK bug fix, security patch, provider-specific behavior, or added API method has been enumerated.

## Validation baseline

Validation is iterative and must use the project's documented commands when available.

1. **Environment:** Record `java -version` and `javac -version`; confirm the validation JDK is 21.
2. **Baseline comparison:** If authorized and still available, record the existing Java 17 build or test result so unrelated failures are not attributed to Java 21.
3. **Wrapper/build tool:** Record Maven or Gradle wrapper versions under the Java 21 environment.
4. **Compile/package:** Run the repository's normal clean verification or build command.
5. **Tests:** Run unit, integration, contract, smoke, and application-startup checks that the repository provides.
6. **Bytecode analysis:** Run JDK 21 `jdeps --jdk-internals` on built application artifacts where practical.
7. **Deprecation analysis:** Run JDK 21 `jdeprscan --for-removal` on built artifacts where practical. Do not use an unsupported `--release 21` argument.
8. **Runtime parity:** Check startup logs, JVM warnings, serialization, charset-sensitive I/O, locale behavior, TLS, agents, and external integrations.
9. **Delivery alignment:** Confirm CI, packaging, container base images, and deployment runtime use the approved Java 21 distribution and configuration.
10. **Post-implementation validation:** When approved source changes were applied, record which selected categories were implemented and rerun the minimum required validation for those categories.

Record command, working directory, exit status, relevant output summary, and whether failure is migration-related, pre-existing, environmental, implementation-related, or unresolved. A build failure caused by the Anteroom Bash sandbox's memory limit is environmental, not proof of Java incompatibility.

## Non-goals and restrictions

- Public API, framework executor, reactive boundary, context propagation, dependency/framework, runtime, and experimental changes are implementable only after their safety-specific warnings, consequences, rollback, validation, and exact-diff approval are displayed.
- Never treat user approval as proof that a Design Sensitive or Experimental change is safe or production-ready.
- Do not recommend preview or incubator features as baseline migration requirements.
- Do not treat modernization completeness as a readiness criterion.
- Do not include optional Java 21 language-feature adoption in compatibility scope. Such adoption is governed by the modernization policy and pattern catalog and is reported separately from readiness.
- Never apply source, build, dependency, CI, container, or runtime configuration changes without explicit, itemized user approval of the exact change set.
- Never commit, push, deploy, modify files outside the current repository, or update external systems unless the user separately requests that additional workflow.

## Readiness statuses

- **READY:** Java 21 configuration is aligned; authorized compile and test validation passed; required delivery and runtime checks passed; no required or high-risk unresolved findings remain.
- **CHANGES REQUIRED:** One or more concrete required changes remain.
- **BLOCKED:** A prerequisite prevents meaningful assessment or validation, such as missing repository files, unavailable JDK 21, or inaccessible inherited configuration.
- **PARTIALLY VALIDATED:** Static assessment completed, but one or more required runtime checks were skipped, unavailable, inconclusive, or environmental.
- **UNSUPPORTED:** The repository is outside version 0.4.0 scope.

An unavailable JDK 21 or build tool is a validation prerequisite blocker, not a required repository change. Record the affected validation checks as `BLOCKED`. The overall report may still be `CHANGES REQUIRED` when separate repository evidence proves changes are required.

## Authoritative sources

Review dates refer to this baseline, not to publication dates.

- Oracle JDK 21 Migration Guide - Preparing for Migration: https://docs.oracle.com/en/java/javase/21/migrate/preparing-migration.html
- Oracle JDK 21 Migration Guide - Significant Changes: https://docs.oracle.com/en/java/javase/21/migrate/significant-changes-jdk-release.html
- Oracle `jdeps` documentation: https://docs.oracle.com/en/java/javase/21/docs/specs/man/jdeps.html
- Oracle `jdeprscan` documentation: https://docs.oracle.com/en/java/javase/21/docs/specs/man/jdeprscan.html
- Gradle Java compatibility matrix: https://docs.gradle.org/current/userguide/compatibility.html
- Apache Maven Compiler Plugin `--release`: https://maven.apache.org/plugins/maven-compiler-plugin/examples/set-compiler-release.html
- Apache Maven Toolchains guide: https://maven.apache.org/guides/mini/guide-using-toolchains.html
- OpenJDK JEP 400, UTF-8 by Default: https://openjdk.org/jeps/400
- OpenJDK JEP 421, Deprecate Finalization for Removal: https://openjdk.org/jeps/421
- OpenJDK JEP 451, dynamic agent loading warnings: https://openjdk.org/jeps/451



# Java 17 to 21 Modernization Policy

**Policy version:** 0.4.0  
**Last reviewed:** 2026-07-26  
**Status:** Draft implementation policy pending organizational Java SME and platform approval

## Purpose

Define how the Java 17 to 21 skills and durable workflow propose, warn about, implement, validate, repair, and report migration changes. Every safety category is implementable after the user reviews the category-specific consequences and approves the exact persisted diff.

Approval accepts disclosed risk; it does not make a risky change safe or prove production compatibility. Readiness remains governed by the baseline, and optional modernization remains separate from readiness.

## User-facing modes

- **assessment:** Static readiness discovery and reporting. No repository code execution or file modification.
- **validate:** Static assessment followed by approved build, test, application, and analysis commands. No file modification.
- **audit:** Static assessment plus complete release-delta and modernization reporting. No file modification.
- **workflow:** Use `fm_java_17_to_21` for persisted discovery, proposal artifacts, exact-diff human gates, implementation, command permission, validation, bounded repair, and final reporting.

The interactive `/java-17-to-21` skill remains a read-only facade. Its legacy
`apply` and `plan` arguments direct users to the durable workflow rather than
recreating approval state inside one chat turn.

## Apply state machine

Use these states in order:

1. `DISCOVERED`: collect static evidence and release-delta coverage.
2. `PROPOSED`: generate categorized changes, exact candidate diffs, warnings, failure consequences, rollback, and validation requirements.
3. `SELECTED`: resolve the workflow `change_scope` (`required`, `safe`,
   `modernize`, or `all`) into stable suggestion IDs. Scope defines the
   proposal; it does not authorize edits.
4. `EDIT_APPROVED`: present the exact combined diff for the selected scope and obtain explicit approval.
5. `APPLIED`: write only the approved diff and produce a revertible change list.
6. `EXECUTION_PERMISSION`: present exact commands and invoke Anteroom's permission flow. The user may choose `Allow Once`, `Allow for Session`, or reject execution.
7. `VALIDATED`: execute only approved commands and record the results.
8. `REPORTED`: produce the final readiness, implementation, validation, consequence, and rollback report.

The workflow may stop at any state. If edits were applied but command execution was rejected, retain the approved edits, mark runtime validation `NOT RUN`, and report no status better than `PARTIALLY VALIDATED`.

## Durable workflow boundaries

- Persist assessment, proposal, validation, and repair evidence in
  repository-root files beginning with the configured `state_prefix`.
- A proposal phase may write only workflow state artifacts. It must not change
  source, build, dependency, CI, container, or runtime files.
- Every source-edit phase must read a complete persisted dossier and unified
  diff, verify that its original hunks are still current, and apply only that
  approved patch.
- Keep human gates as top-level workflow steps. Do not place a human gate
  inside a native loop or parallel branch.
- Implement the bounded repair cycle as two explicit
  propose-repair → human-gate → apply → revalidate rounds. This preserves
  resumability and exact approval for each repair.
- Workflow state-artifact permission, exact source-diff approval, and
  repository-command permission are distinct authorization boundaries.
- A stopped or expired gate never implies approval. Resume only through
  Anteroom's recorded workflow decision and approval commands.

## Required proposal fields

Every proposed change must include:

- stable suggestion identifier
- category and pattern
- classification and safety class
- file and line evidence
- evidence type and confidence
- exact proposed diff
- implementation scope and affected modules
- assumptions
- expected benefit
- consequences if the assumptions are wrong or validation fails
- rollback procedure
- required validation commands/checks
- preview/incubator flags and portability impact when applicable

## Safety classes and required warnings

All four classes are implementable after exact approval. The class controls warning strength and validation, not eligibility.

### Safe Mechanical

Narrow, local changes with low expected semantic risk.

Required warning:

- compilation or focused tests may still fail
- formatting, string, or configuration semantics may differ if the detected context was incomplete
- the exact diff should be easy to revert

These may be batch-selected and approved together.

### Review Required

Changes dependent on local intent or behavior, including ordering, null handling, serialization, control flow, configuration inheritance, or dependency compatibility.

Required warning:

- the change may compile while altering application behavior
- tests may need updates or may not cover the affected assumption
- dependency or plugin changes may introduce transitive changes, downloads, or new configuration requirements
- failure requires reverting the change or completing follow-up remediation

These may be selected individually, by category, or through `all proposed changes`, but their warning and consequences must appear in the final combined approval.

### Design Sensitive

Changes that may affect public APIs, concurrency, framework integration, lifecycle, security/transaction/request/logging context, runtime flags, native integration, or production behavior.

Required warning:

- downstream consumers or framework-managed behavior may break
- compilation and unit tests may pass while integration, performance, deadlock, data consistency, or context-propagation failures remain
- the change may require coordinated changes across modules or services
- rollback and representative integration/runtime validation are mandatory

These are implementable only after the exact blast radius, known unknowns, failure consequences, rollback, and required integration/runtime validation are displayed.

### Experimental

Preview, incubating, or trial-oriented language/API/runtime work.

Required warning:

- preview/incubator flags may be required at compile and runtime
- the API or language feature may change or disappear in later JDKs
- IDE, build, library, runtime, or vendor support may be incomplete
- production supportability and future upgrades may become harder
- failure may require removing the feature and restoring the previous implementation

These are implementable only after explicit opt-in. Selecting `all proposed changes` counts as opt-in only when the combined approval clearly states that experimental changes and their flags are included.

## Selection and edit approval

Present all proposed changes grouped by classification and safety class. The
workflow's `change_scope` may select:

- `required`: required migration changes only
- `safe`: required plus Safe Mechanical changes
- `modernize`: required plus non-experimental modernization
- `all`: every responsible proposed change, including explicitly disclosed
  Design Sensitive and Experimental work

After selection, persist one exact combined diff containing only the selected
changes and a matching dossier. Repeat category warnings and aggregate
consequences, affected modules, validation, and rollback. The human gate must
name those artifacts and obtain explicit approval before source files are
written.

Approval of a category does not authorize unlisted future findings. Approval of a diff authorizes only that displayed edit set.

## Command execution permission

After edits, show every proposed command with:

- exact command and working directory
- purpose
- expected file/process/network side effects
- dependency-download possibility
- affected suggestion IDs
- stop conditions

Then invoke Anteroom's execution permission flow:

- **Allow Once:** authorize the displayed execution request once.
- **Allow for Session:** allow execution for the current Anteroom session. Even when the tool permission persists, this pack must remain inside the current repository and migration workflow and must explain materially different or expanded commands before running them.
- **Reject:** do not execute commands; report validation as not run or blocked.

Do not request `Allow Always` as part of the normal migration workflow. A persistent configuration change is outside this pack unless the user explicitly requests it.

## Execution and failure behavior

- Apply changes in reversible batches where dependencies between edits allow it.
- Record a before/after change list and rollback instructions for every batch.
- Stop the affected batch when an approved command fails.
- Classify the failure as migration-related, implementation-related, pre-existing, environmental, permission-related, or unresolved.
- Do not silently broaden the diff or command set to repair a failure.
- Propose the repair as a new categorized change with its own consequences and approval.
- Never commit, push, deploy, or update external systems unless the user separately and explicitly requests that additional workflow.
- Never modify files outside the current repository.
- Redact secrets and sensitive values from proposals and reports.

## Reporting requirements

The final report must separate:

- readiness findings
- modernization findings
- selected and rejected changes
- warnings and accepted consequences by safety class
- exact changes applied
- commands and permission scope used
- validation results and failures
- rollback instructions
- remaining findings and checks not performed

A repository may be `READY` for Java 21 while optional modernization remains, and may be `PARTIALLY VALIDATED` after approved changes when command execution or required runtime validation was not completed.


# Java 17 to 21 Pattern Catalog

**Catalog version:** 0.4.0  
**Last reviewed:** 2026-07-26  
**Status:** Initial modernization pattern catalog pending organizational Java SME approval

## Purpose

This catalog defines actionable pattern families, detection heuristics, evidence requirements, default classifications, safety classes, implementation approval, pattern-specific consequences, and post-change validation expectations for the `/java-17-to-21` skill. Use it together with `java-17-to-21-release-delta`; the release delta defines completeness, while this catalog defines actionable findings.

## Pattern catalog template

Each pattern entry should be interpreted with these fields:

- **Pattern family:** Configuration, dependency compatibility, runtime risk, language modernization, API modernization, concurrency modernization, delivery/runtime alignment, or design-sensitive.
- **Pattern name:** Concise identifier such as `virtual-thread-candidate`, `threadlocal-risk`, or `switch-pattern-candidate`.
- **Intent:** Why the pattern exists and what migration or modernization question it helps answer.
- **Detection heuristics:** Source constructs, configuration keys, dependency names, command evidence, or file patterns to match.
- **Evidence to capture:** File path, line number, matched text, surrounding context, and whether the evidence is static or validated.
- **Why it matters:** Migration risk, modernization value, or runtime concern associated with the pattern.
- **Default classification:** REQUIRED, ADVISORY, UNRESOLVED, MODERNIZATION_CANDIDATE, or EXPERIMENTAL.
- **Safety class:** Safe Mechanical, Review Required, Design Sensitive, or Experimental.
- **Implementation approval:** Every pattern is implementable after the warning and approval required by its safety class. State any additional pattern-specific opt-in or command permission.
- **Validation required after change:** Compile, unit tests, integration tests, startup checks, runtime checks, warning review, or performance/concurrency validation.
- **Non-goals / cautions:** What should not be inferred from a match. Use these cautions plus the safety-class policy to state concrete failure consequences and rollback requirements before approval.

When a pattern relates to a JDK change, also capture its release/JEP provenance and distinguish final, preview, incubator, deprecated, and earlier-language-modernization features.

## Configuration patterns

### java-release-still-17

- **Pattern family:** Configuration
- **Intent:** Detect repositories still configured to emit Java 17 bytecode when the migration target is Java 21.
- **Detection heuristics:** `maven.compiler.release`, compiler `<release>`, `<source>`, `<target>`, `java.version`, Gradle `JavaLanguageVersion.of(17)`, `sourceCompatibility = 17`, `targetCompatibility = 17`, `options.release = 17`.
- **Evidence to capture:** Build file path, line number, property/plugin block, inherited source if known.
- **Why it matters:** The project is not configured for Java 21 output.
- **Default classification:** REQUIRED
- **Safety class:** Safe Mechanical for local release-property edits; Review Required when inherited or convention-managed.
- **Auto-apply policy:** implementable after exact approval; inherited or convention-managed values require Review Required warnings and coordinated validation.
- **Validation required after change:** Clean compile/build and test on JDK 21.
- **Non-goals / cautions:** Do not change inherited parent, convention plugin, or enterprise baseline values without explicit approval.

### wrapper-runtime-mismatch

- **Pattern family:** Configuration
- **Intent:** Detect Maven/Gradle wrapper or runtime incompatibility risks.
- **Detection heuristics:** `gradle-wrapper.properties`, Maven wrapper files, CI image/tool versions, plugin versions.
- **Evidence to capture:** Wrapper version, distribution URL, CI/runtime Java version, file path and line.
- **Why it matters:** Builds can fail before source compatibility is tested.
- **Default classification:** REQUIRED when below known support level; UNRESOLVED when support cannot be established.
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose distribution downloads, checksum/provenance, plugin compatibility, and rollback consequences.
- **Validation required after change:** Wrapper version command and clean build on JDK 21.
- **Non-goals / cautions:** Do not recommend uncontrolled jumps to the newest major version.

## Dependency and runtime risk patterns

### dependency-compatibility-unresolved

- **Pattern family:** Dependency compatibility
- **Intent:** Track frameworks, plugins, annotation processors, bytecode libraries, agents, drivers, and native integrations whose Java 21 support is not proven locally.
- **Detection heuristics:** Direct versions, managed versions, BOMs, parent POMs, Gradle platforms, version catalogs, build plugins.
- **Evidence to capture:** Dependency or plugin coordinate, version source, management source, file path and line.
- **Why it matters:** Dependencies often cause migration failures before source code does.
- **Default classification:** UNRESOLVED
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval when an authoritative target version and concrete diff are available; disclose transitive changes and downloads.
- **Validation required after change:** Compile, tests, startup, and library-specific compatibility checks.
- **Non-goals / cautions:** Do not label a dependency incompatible solely because it is old.

### reflective-access-risk

- **Pattern family:** Runtime risk
- **Intent:** Detect strong encapsulation and internal API risk.
- **Detection heuristics:** `sun.*`, most `com.sun.*`, `jdk.internal.*`, `Unsafe`, `--add-opens`, `--add-exports`, `--illegal-access`.
- **Evidence to capture:** Matched package/API/flag, owning dependency or code path if available.
- **Why it matters:** Runtime-only failures can appear on newer JDKs.
- **Default classification:** UNRESOLVED or REQUIRED when validation proves failure.
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose runtime-access failure risk and retain rollback for flags or source changes.
- **Validation required after change:** JDK 21 tests, startup, `jdeps --jdk-internals`, and warning review.
- **Non-goals / cautions:** `--add-opens` alone does not prove the flag can be removed.

## Language modernization patterns

### instanceof-pattern-cleanup

- **Pattern family:** Language modernization
- **Intent:** Detect local `instanceof` plus cast cleanup opportunities.
- **Detection heuristics:** `if (x instanceof Type) { Type y = (Type) x; ... }`
- **Evidence to capture:** Source file, line range, variable names, local usage.
- **Why it matters:** Pattern variables reduce boilerplate without changing public contracts.
- **Release provenance:** Earlier language modernization; pattern matching for `instanceof` was final in Java 16.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Safe Mechanical
- **Auto-apply policy:** implementable after Safe Mechanical warning and exact approval.
- **Validation required after change:** Compile and relevant unit tests.
- **Non-goals / cautions:** Do not alter control flow or variable scope assumptions.

### switch-pattern-candidate

- **Pattern family:** Language modernization
- **Intent:** Detect switch modernization opportunities.
- **Detection heuristics:** Type-dispatch chains that could use type or record patterns, guarded type cases, and null-handling branches. Report ordinary switch-expression cleanup separately as earlier language modernization.
- **Evidence to capture:** Source file, branch structure, involved types.
- **Why it matters:** Java 21 finalizes pattern matching for switch and can simplify type dispatch.
- **Release provenance:** JEP 441, final in Java 21.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose exhaustiveness, null, branch-order, and control-flow consequences.
- **Validation required after change:** Compile, unit tests, branch coverage review.
- **Non-goals / cautions:** Do not rewrite complex control flow without Review Required warnings, exact approval, branch-behavior consequences, rollback, and tests.

### record-candidate

- **Pattern family:** Language modernization
- **Intent:** Detect immutable data carriers that may be record candidates.
- **Detection heuristics:** Final fields, canonical constructor, accessors, `equals`, `hashCode`, `toString`.
- **Evidence to capture:** Source file, class signature, constructors, public API usage.
- **Why it matters:** Records can reduce boilerplate but change API shape and serialization behavior.
- **Release provenance:** Earlier language modernization; record classes were final in Java 16. This is distinct from Java 21 record patterns.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose public API, serialization, reflection, framework binding, constructor, and accessor consequences.
- **Validation required after change:** Compile, serialization/API compatibility tests, framework binding tests.
- **Non-goals / cautions:** Do not convert public API or framework-bound classes without Design Sensitive warning, downstream-impact disclosure, exact approval, and integration validation.

### sealed-hierarchy-candidate

- **Pattern family:** Language modernization
- **Intent:** Identify type hierarchies that may benefit from sealed classes or interfaces.
- **Detection heuristics:** Closed implementation sets, package-private constructors, discriminator logic.
- **Evidence to capture:** Hierarchy root, implementors, external extension risk.
- **Why it matters:** Sealed hierarchies can document and enforce domain boundaries.
- **Release provenance:** Earlier language modernization; sealed classes and interfaces were final in Java 17.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose extension, proxying, reflection, and downstream compilation consequences.
- **Validation required after change:** Compile, API compatibility, framework proxying/binding tests.
- **Non-goals / cautions:** Do not restrict extension points without Design Sensitive warning, downstream-impact disclosure, exact approval, and compatibility validation.

### text-block-candidate

- **Pattern family:** Language modernization
- **Intent:** Detect multiline string literals that could become text blocks.
- **Detection heuristics:** String concatenation with repeated newlines, SQL/JSON/XML literals.
- **Evidence to capture:** Source file, literal, escaping behavior.
- **Why it matters:** Text blocks improve readability.
- **Release provenance:** Earlier language modernization; text blocks were final in Java 15.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Safe Mechanical when exact string semantics are preserved.
- **Auto-apply policy:** implementable after Safe Mechanical warning and exact approval with before/after string-semantics comparison.
- **Validation required after change:** Compile and string-sensitive tests.
- **Non-goals / cautions:** Do not alter whitespace, escaping, or trailing-newline semantics.

### record-pattern-candidate

- **Pattern family:** Language modernization
- **Intent:** Detect Java 21 record deconstruction opportunities independently of record-class conversion.
- **Detection heuristics:** Type tests followed by record component accessor calls, nested record traversal, or switch type cases over record classes.
- **Evidence to capture:** Record declaration, source file and line range, accessor chain, null assumptions, and exhaustiveness requirements.
- **Why it matters:** Record patterns can make data navigation declarative without converting existing classes into records.
- **Release provenance:** JEP 440, final in Java 21.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose null, nesting, exhaustiveness, and generic inference consequences.
- **Validation required after change:** Compile, unit tests, null-path tests, and switch exhaustiveness review.
- **Non-goals / cautions:** Do not confuse record patterns with record declarations or change a class into a record.

### preview-language-feature-use

- **Pattern family:** Language modernization
- **Intent:** Detect use or proposed adoption of Java 21 preview language features.
- **Detection heuristics:** `--enable-preview`, string-template syntax, unnamed patterns/variables, or unnamed classes and instance main methods.
- **Evidence to capture:** Build/runtime preview flags, source construct, owning module, and deployment compatibility.
- **Why it matters:** JDK 21 includes String Templates (JEP 430), Unnamed Patterns and Variables (JEP 443), and Unnamed Classes and Instance Main Methods (JEP 445) as previews, not permanent baseline features.
- **Release provenance:** JEPs 430, 443, and 445, preview in Java 21.
- **Default classification:** EXPERIMENTAL
- **Safety class:** Experimental
- **Auto-apply policy:** implementable after Experimental explicit opt-in and exact approval; include compile/runtime preview flags and future-JDK removal or change consequences.
- **Validation required after change:** Explicit opt-in, preview-enabled compile/test/runtime, and future-JDK portability review.
- **Non-goals / cautions:** Never introduce or recommend a preview feature as a Java 21 readiness requirement.

## API modernization patterns

### sequenced-collection-candidate

- **Pattern family:** API modernization
- **Intent:** Detect opportunities to use Java 21 sequenced collection APIs.
- **Detection heuristics:** Manual first/last/reverse helpers, ordered collection assumptions.
- **Evidence to capture:** Source file, collection type, ordering contract.
- **Why it matters:** Java 21 adds explicit ordered collection abstractions.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose encounter-order, mutability, view, and compatibility consequences.
- **Validation required after change:** Compile and behavioral tests for ordering.
- **Non-goals / cautions:** Do not assume every `List` or `LinkedHashMap` usage should change.

### stream-optional-cleanup

- **Pattern family:** API modernization
- **Intent:** Detect verbose stream or Optional idioms that could be simplified.
- **Detection heuristics:** Manual null checks around Optional, redundant stream collections, verbose first/last helpers.
- **Evidence to capture:** Source file, expression, expected behavior.
- **Why it matters:** Local cleanup can improve clarity after the migration target is established.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose stream laziness, ordering, parallelism, null behavior, and side-effect consequences.
- **Validation required after change:** Compile and affected unit tests.
- **Non-goals / cautions:** Where laziness, ordering, parallelism, or side effects matter, disclose the behavior change and tests before exact approval.

### javadoc-snippet-candidate

- **Pattern family:** API modernization
- **Intent:** Identify Javadoc examples that may benefit from the JDK 18 `@snippet` tag.
- **Detection heuristics:** Large `<pre>{@code ...}</pre>` blocks, copied example code, or external snippet files referenced manually.
- **Evidence to capture:** Documentation file and line range, source of the example, and current doclint behavior.
- **Why it matters:** `@snippet` can improve example maintainability and validation.
- **Release provenance:** JEP 413, final in JDK 18 tooling.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose doclint and generated-documentation consequences.
- **Validation required after change:** Javadoc generation and doclint.
- **Non-goals / cautions:** Do not change executable behavior or claim examples are tested unless a test verifies them.

### kem-api-candidate

- **Pattern family:** API modernization
- **Intent:** Identify relevant custom key-encapsulation code that may use the Java 21 KEM API.
- **Detection heuristics:** Cryptographic key encapsulation, provider-specific KEM abstractions, or protocol implementations using asymmetric wrapping for symmetric keys.
- **Evidence to capture:** Algorithm/provider, protocol boundary, compliance requirements, and existing tests.
- **Why it matters:** Java 21 provides a standard KEM API, but cryptographic changes require specialized review.
- **Release provenance:** JEP 452, final in Java 21.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning, security-specific consequences, and exact approval.
- **Validation required after change:** Security review, provider interoperability, known-answer tests, and protocol integration tests.
- **Non-goals / cautions:** Do not redesign cryptography or select algorithms without Design Sensitive warning, security-review consequences, exact approval, interoperability tests, and rollback.

## Concurrency modernization patterns

### virtual-thread-candidate

- **Pattern family:** Concurrency modernization
- **Intent:** Identify areas that may benefit from virtual threads.
- **Detection heuristics:** `new Thread`, `Executors.newFixedThreadPool`, blocking I/O tasks, request-per-thread patterns.
- **Evidence to capture:** Executor ownership, task type, blocking behavior, framework involvement.
- **Why it matters:** Java 21 introduces major concurrency opportunities.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose throttling, ownership, pinning, context, integration, and performance consequences.
- **Validation required after change:** Integration/runtime tests, performance/concurrency validation, startup checks.
- **Non-goals / cautions:** Do not rewrite framework-managed executors or throttling designs without Design Sensitive warning, exact approval, and representative runtime validation.

### threadlocal-risk

- **Pattern family:** Concurrency modernization
- **Intent:** Detect `ThreadLocal` and `InheritableThreadLocal` usage that may interact badly with concurrency model changes.
- **Detection heuristics:** `ThreadLocal`, `InheritableThreadLocal`, request/security/logging context holders.
- **Evidence to capture:** Source file, owning context, lifecycle, cleanup behavior.
- **Why it matters:** Context propagation assumptions can break with executor or virtual-thread changes.
- **Default classification:** UNRESOLVED or MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose context loss/leakage, lifecycle, security, tracing, logging, and transaction consequences.
- **Validation required after change:** Integration tests for request, security, tracing, logging, and transaction context.
- **Non-goals / cautions:** Do not modify context propagation without Design Sensitive warning, exact approval, and context-specific integration validation.

### pinning-risk

- **Pattern family:** Concurrency modernization
- **Intent:** Identify pinning-sensitive synchronized or monitor-heavy regions around blocking work.
- **Detection heuristics:** `synchronized` blocks/methods, blocking I/O, waits, locks around external calls.
- **Evidence to capture:** Source file, lock region, blocking call evidence.
- **Why it matters:** Pinning can reduce benefits of virtual threads.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose deadlock, race, ordering, throughput, and correctness consequences.
- **Validation required after change:** Concurrency tests and runtime profiling.
- **Non-goals / cautions:** Do not change locking semantics without Design Sensitive warning, exact approval, rollback, concurrency tests, and profiling.

### structured-concurrency-trial

- **Pattern family:** Concurrency modernization
- **Intent:** Identify fan-out/fan-in logic that could be trialed with structured concurrency.
- **Detection heuristics:** sibling `CompletableFuture` orchestration, manual thread joins, coordinated subtasks.
- **Evidence to capture:** Source file, cancellation/error handling, task ownership.
- **Why it matters:** Structured concurrency can simplify lifecycle management but remains trial-oriented.
- **Default classification:** EXPERIMENTAL
- **Safety class:** Experimental
- **Auto-apply policy:** implementable after Experimental explicit opt-in and exact approval; include preview flags plus cancellation, error propagation, observability, and future-JDK consequences.
- **Validation required after change:** Explicit opt-in prototype, integration tests, cancellation/error-path validation.
- **Non-goals / cautions:** Never present structured concurrency as a baseline migration requirement.

### scoped-value-trial

- **Pattern family:** Concurrency modernization
- **Intent:** Relate `ThreadLocal` context usage to optional scoped-value experiments.
- **Detection heuristics:** Read-mostly `ThreadLocal` values with bounded call lifetimes, especially around virtual-thread candidates.
- **Evidence to capture:** Context ownership, mutation, inheritance, cleanup, framework integration, and call boundaries.
- **Why it matters:** Scoped Values were previewed in Java 21 as an alternative for some immutable context-sharing cases.
- **Release provenance:** JEP 446, preview in Java 21.
- **Default classification:** EXPERIMENTAL
- **Safety class:** Experimental
- **Auto-apply policy:** implementable after Experimental explicit opt-in and exact approval; include preview flags plus context semantics, framework support, and future-JDK consequences.
- **Validation required after change:** Explicit prototype opt-in and context, security, tracing, transaction, and concurrency tests.
- **Non-goals / cautions:** Do not replace `ThreadLocal` without Experimental opt-in and context-specific validation; do not assume scoped values fit mutable context.

### foreign-function-memory-use

- **Pattern family:** Concurrency modernization
- **Intent:** Detect existing or proposed use of the Java 21 preview Foreign Function & Memory API.
- **Detection heuristics:** `java.lang.foreign`, `Arena`, `MemorySegment`, linker calls, native bindings, or preview flags.
- **Evidence to capture:** Native library, lifecycle/ownership, platform targets, preview flags, and JNI/JNA alternatives.
- **Why it matters:** The API was still preview in Java 21 and is platform- and lifecycle-sensitive.
- **Release provenance:** JEP 442, third preview in Java 21.
- **Default classification:** EXPERIMENTAL
- **Safety class:** Experimental
- **Auto-apply policy:** implementable after Experimental explicit opt-in and exact approval; include preview flags plus native memory, lifecycle, ABI, platform, and future-JDK consequences.
- **Validation required after change:** Native integration tests, memory-lifetime tests, platform matrix, and explicit preview opt-in.
- **Non-goals / cautions:** Do not replace JNI/JNA without Experimental opt-in, exact approval, native-lifecycle analysis, and platform validation.

### vector-api-use

- **Pattern family:** Concurrency modernization
- **Intent:** Detect use or performance-sensitive opportunities for the incubating Vector API.
- **Detection heuristics:** `jdk.incubator.vector`, explicit incubator modules, or measured scalar numeric hotspots.
- **Evidence to capture:** Module flags, supported architectures, benchmark evidence, and fallback behavior.
- **Why it matters:** The Vector API remained incubating in Java 21 and should be driven by measurement.
- **Release provenance:** JEP 448, sixth incubator in Java 21.
- **Default classification:** EXPERIMENTAL
- **Safety class:** Experimental
- **Auto-apply policy:** implementable after Experimental explicit opt-in and exact approval; include incubator modules plus numerical, architecture, benchmark, and future-JDK consequences.
- **Validation required after change:** JMH or equivalent benchmarks, numerical correctness, architecture coverage, and fallback tests.
- **Non-goals / cautions:** Do not propose vectorization without a measured hotspot.

## Platform and runtime delta patterns

### reflection-behavior-risk

- **Pattern family:** Runtime risk
- **Intent:** Identify reflection-heavy behavior that may be affected by the JDK 18 core-reflection reimplementation.
- **Detection heuristics:** Intensive `java.lang.reflect`/method-handle use, generated accessors, serialization frameworks, agents, and reflection performance assumptions.
- **Evidence to capture:** API/framework, code path, encapsulation flags, and Java 21 runtime evidence.
- **Why it matters:** JEP 416 changed the reflection implementation while preserving API contracts; edge behavior and performance still require validation.
- **Release provenance:** JEP 416, final in JDK 18.
- **Default classification:** UNRESOLVED
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose reflection access, serialization, framework, startup, and performance consequences.
- **Validation required after change:** Framework tests, startup, serialization, reflection access, and performance checks where relevant.
- **Non-goals / cautions:** Do not infer incompatibility from reflection use alone.

### resolver-spi-candidate

- **Pattern family:** Runtime risk
- **Intent:** Identify custom hostname/address-resolution integrations affected by or suitable for the InetAddress resolver SPI.
- **Detection heuristics:** Custom DNS libraries, hosts-file overrides, `InetAddress` wrappers, or provider configuration.
- **Evidence to capture:** Resolver ownership, caching, security policy, deployment configuration, and failure behavior.
- **Why it matters:** JDK 18 added a standard resolver SPI, but network semantics are environment-sensitive.
- **Release provenance:** JEP 418, final in JDK 18.
- **Default classification:** MODERNIZATION_CANDIDATE or UNRESOLVED
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose DNS, caching, failover, security, and deployment consequences.
- **Validation required after change:** DNS, caching, failover, security, and deployment integration tests.
- **Non-goals / cautions:** Do not replace production name resolution without Design Sensitive warning, exact approval, rollback, and environment-specific validation.

### generational-zgc-candidate

- **Pattern family:** Runtime risk
- **Intent:** Detect ZGC use or workloads that may warrant a Generational ZGC evaluation.
- **Detection heuristics:** `-XX:+UseZGC`, latency-sensitive workloads, allocation-heavy services, and GC tuning files.
- **Evidence to capture:** JVM flags, heap/container limits, latency objectives, GC logs, and baseline performance.
- **Why it matters:** Java 21 added Generational ZGC, but collector selection and tuning require measurement.
- **Release provenance:** JEP 439, final in JDK 21.
- **Default classification:** MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose latency, throughput, memory, container, operational, and rollback consequences.
- **Validation required after change:** Representative load tests, GC-log analysis, latency/throughput comparison, and rollback plan.
- **Non-goals / cautions:** Do not change garbage collectors or production JVM flags without Design Sensitive warning, exact approval, performance evidence, and rollback.

### platform-architecture-risk

- **Pattern family:** Delivery/runtime alignment
- **Intent:** Cover the new Linux/RISC-V port and deprecation of the Windows 32-bit x86 port.
- **Detection heuristics:** Architecture-specific images, native libraries, JNI/JNA artifacts, Windows x86 runners, or Linux RISC-V deployment targets.
- **Evidence to capture:** Target OS/architecture matrix, container base images, native artifacts, CI runners, and vendor support.
- **Why it matters:** JDK 19 added Linux/RISC-V support and JDK 21 deprecated Windows x86-32 for removal.
- **Release provenance:** JEP 422 and JEP 449.
- **Default classification:** UNRESOLVED or REQUIRED when an approved target is unsupported.
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose native artifact, target availability, packaging, CI, and deployment consequences.
- **Validation required after change:** Target-platform build, packaging, startup, and native integration tests.
- **Non-goals / cautions:** Do not infer target architectures from the developer workstation alone.

### simple-web-server-use

- **Pattern family:** Delivery/runtime alignment
- **Intent:** Detect intentional use or inappropriate production use of the JDK simple web server.
- **Detection heuristics:** `jwebserver`, `com.sun.net.httpserver.SimpleFileServer`, or repository scripts/documentation invoking them.
- **Evidence to capture:** Script or source location, environment, network binding, and intended purpose.
- **Why it matters:** JEP 408 provides a testing/prototyping tool, not a production server recommendation.
- **Release provenance:** JEP 408, final in JDK 18 tooling/API.
- **Default classification:** ADVISORY or UNRESOLVED
- **Safety class:** Review Required
- **Auto-apply policy:** implementable after Review Required warning and exact approval; disclose network exposure and non-production suitability consequences.
- **Validation required after change:** Script/documentation review and environment-specific smoke test.
- **Non-goals / cautions:** Never recommend the simple web server for production serving.

## Design-sensitive patterns

### public-api-change-risk

- **Pattern family:** Design-sensitive
- **Intent:** Flag candidates that would alter external contracts.
- **Detection heuristics:** Public classes/methods, exported packages, DTOs, framework-bound APIs.
- **Evidence to capture:** Public signature, consumers if known, serialization or reflection use.
- **Why it matters:** API changes can break callers even when source compiles.
- **Default classification:** EXPERIMENTAL or MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose binary/source compatibility, serialization, reflection, and downstream consumer consequences.
- **Validation required after change:** API compatibility and downstream integration testing.
- **Non-goals / cautions:** Do not change public API signatures without Design Sensitive warning, downstream-consumer impact, exact approval, compatibility checks, and rollback.

### context-propagation-risk

- **Pattern family:** Design-sensitive
- **Intent:** Detect transaction, security, request, tracing, or logging context propagation assumptions.
- **Detection heuristics:** ThreadLocal context holders, framework context APIs, MDC, security context, transactions.
- **Evidence to capture:** Source file, context type, lifecycle and cleanup.
- **Why it matters:** Context propagation can change under executor or concurrency modernization.
- **Default classification:** UNRESOLVED
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose transaction, security, request, tracing, logging, lifecycle, and data-consistency consequences.
- **Validation required after change:** Integration and runtime context validation.
- **Non-goals / cautions:** Do not modify context propagation without Design Sensitive warning, exact approval, and transaction/security/request/tracing/logging validation.

### framework-executor-risk

- **Pattern family:** Design-sensitive
- **Intent:** Detect framework-managed execution boundaries.
- **Detection heuristics:** Spring executors, scheduled tasks, async annotations, reactive schedulers, container-managed pools.
- **Evidence to capture:** Framework configuration, executor ownership, lifecycle.
- **Why it matters:** Framework-managed concurrency changes can alter runtime behavior.
- **Default classification:** UNRESOLVED or MODERNIZATION_CANDIDATE
- **Safety class:** Design Sensitive
- **Auto-apply policy:** implementable after Design Sensitive warning and exact approval; disclose framework lifecycle, scheduler, backpressure, reactive-boundary, and runtime consequences.
- **Validation required after change:** Framework integration and startup/runtime tests.
- **Non-goals / cautions:** Do not rewrite framework-managed executors or reactive boundaries without Design Sensitive warning, exact approval, framework-specific validation, and rollback.


# Java 17 to 21 Release Delta

**Delta version:** 0.4.0  
**Last reviewed:** 2026-07-26  
**Status:** Complete JEP-level inventory pending organizational Java SME approval

## Purpose

Use this inventory to ensure that a Java 17 to 21 assessment considers every JEP integrated since JDK 17, not only common source-modernization examples. The inventory follows the OpenJDK consolidated JDK 17-to-21 list, which omits superseded intermediate previews and retains the final Java 21 state.

This is a **JEP-level completeness boundary**. It does not claim to enumerate every added method, resolved bug, provider-specific change, or security patch. Use the Oracle migration guide, release notes, API removals/deprecations, `jdeprscan`, and application-specific testing for those surfaces.

## Required coverage behavior

For every entry below, report one coverage disposition:

- `APPLICABLE_FINDING`: repository evidence makes the item relevant and a finding was created.
- `APPLICABLE_NO_FINDING`: the item was checked and no risk or opportunity was detected.
- `NOT_APPLICABLE`: the repository does not use the affected platform, API, tool, or behavior.
- `UNRESOLVED`: available evidence cannot establish applicability.
- `NOT_CHECKED`: the check was outside the approved or available validation surface; explain why.

Preview and incubator entries are never Java 21 readiness requirements merely because they exist. Detect existing use and optional trial opportunities, classify them `EXPERIMENTAL`, and allow implementation only after explicit opt-in, failure consequences, exact-diff approval, and preview/incubator-aware validation.

## Final additions since JDK 17

| JEP | Release area | Change | Default assessment treatment |
| --- | --- | --- | --- |
| 400 | JDK 18 / I/O | UTF-8 by Default | Check implicit default-charset dependencies and runtime parity. |
| 408 | JDK 18 / Tooling | Simple Web Server | Usually not applicable; identify only repository scripts or documentation that could intentionally use `jwebserver`. Never recommend it for production serving. |
| 413 | JDK 18 / Javadoc | Code Snippets in Java API Documentation | Detect legacy embedded Javadoc examples as optional documentation modernization. |
| 416 | JDK 18 / Reflection | Reimplement Core Reflection with Method Handles | Review reflection-heavy frameworks, serialization, agents, and performance/behavior assumptions. |
| 418 | JDK 18 / Networking | Internet-Address Resolution SPI | Review custom DNS/host-resolution code or resolver-provider requirements. |
| 422 | JDK 19 / HotSpot | Linux/RISC-V Port | Check only when delivery targets Linux RISC-V or native artifacts constrain architecture. |
| 431 | JDK 21 / Collections | Sequenced Collections | Detect ordered-collection helpers and API-modernization candidates. |
| 439 | JDK 21 / GC | Generational ZGC | Review ZGC flags, performance requirements, container memory limits, and runtime validation needs. |
| 440 | JDK 21 / Language | Record Patterns | Detect record deconstruction opportunities separately from record-class conversion. |
| 441 | JDK 21 / Language | Pattern Matching for `switch` | Detect type-dispatch and pattern-switch opportunities separately from older switch expressions. |
| 444 | JDK 21 / Concurrency | Virtual Threads | Audit executor ownership, blocking work, throttling, pinning, and context propagation. |
| 451 | JDK 21 / Serviceability | Prepare to Disallow Dynamic Agent Loading | Detect agent attachment and validate warnings and startup configuration. |
| 452 | JDK 21 / Cryptography | Key Encapsulation Mechanism API | Report only when cryptographic key encapsulation or provider integration is relevant; require security review. |

## Preview and incubator state in JDK 21

| JEP | Feature | Required treatment |
| --- | --- | --- |
| 430 | String Templates (Preview) | Detect `--enable-preview` and source use; implement only after `EXPERIMENTAL` opt-in and preview/future-portability warning. |
| 442 | Foreign Function & Memory API (Third Preview) | Detect preview API/native-memory use; implement only after `EXPERIMENTAL` opt-in and native-lifecycle/platform warning. |
| 443 | Unnamed Patterns and Variables (Preview) | Detect existing preview use or optional trials; implement only after `EXPERIMENTAL` opt-in and portability warning. |
| 445 | Unnamed Classes and Instance Main Methods (Preview) | Usually not applicable to production repositories; implement only after `EXPERIMENTAL` opt-in and supportability warning. |
| 446 | Scoped Values (Preview) | Relate to `ThreadLocal`/context-propagation audits; implement only after `EXPERIMENTAL` opt-in and context/framework warning. |
| 448 | Vector API (Sixth Incubator) | Detect incubator module/API use and performance-sensitive opportunities; implement only after `EXPERIMENTAL` opt-in and benchmark/architecture warning. |
| 453 | Structured Concurrency (Preview) | Detect fan-out/fan-in trials; implement only after `EXPERIMENTAL` opt-in and cancellation/error/future-portability warning. |

## Deprecations

| JEP | Change | Default assessment treatment |
| --- | --- | --- |
| 421 | Deprecate Finalization for Removal | Detect custom finalizers and finalization dependencies; require lifecycle review. |
| 449 | Deprecate the Windows 32-bit x86 Port for Removal | Check delivery architecture only when Windows x86-32 is relevant. |

## Earlier language features that remain useful modernization targets

The pattern catalog may also suggest features finalized before or in JDK 17 because Java 17 code can still contain older idioms. Label them `EARLIER_LANGUAGE_MODERNIZATION`, not as new JDK 18-21 additions:

- switch expressions: final in Java 14
- text blocks: final in Java 15
- pattern matching for `instanceof`: final in Java 16
- record classes: final in Java 16
- sealed classes and interfaces: final in Java 17

## Authoritative sources

- OpenJDK JEPs integrated since JDK 17: https://openjdk.org/projects/jdk/21/jeps-since-jdk-17
- OpenJDK JDK 21 feature list: https://openjdk.org/projects/jdk/21/
- Oracle JDK 21 Migration Guide, Significant Changes: https://docs.oracle.com/en/java/javase/21/migrate/significant-changes-jdk-release.html
- Oracle JDK 21 Migration Guide, Removed Tools and Components: https://docs.oracle.com/en/java/javase/21/migrate/removed-tools-and-components.html
- Oracle JDK 21 Migration Guide, Removed APIs: https://docs.oracle.com/en/java/javase/21/migrate/removed-apis.html


---
name: java-17-to-21
description: Provide an interactive Java 17-to-21 assessment, audit, validation, or workflow handoff for a local Maven or Gradle repository. Use for one-off analysis or when a user asks how to start the durable fm_java_17_to_21 migration workflow.
---

# Java 17 to 21 interactive facade

Follow the `java-17-to-21-baseline`, `java-17-to-21-release-delta`,
`java-17-to-21-modernization-policy`, and
`java-17-to-21-pattern-catalog` instructions supplied by this pack.

Invocation arguments: {args}

## Modes

- Default or `assessment`: invoke the behavior of `java-17-to-21-assess`
  without creating workflow state files unless the user requests them.
- `audit`: perform the assessment plus complete release-delta and
  modernization coverage. Do not modify files or execute repository code.
- `validate`: perform the assessment, show exact validation commands and
  side effects, then use `ask_user` before invoking `bash`. Do not edit.
- `workflow`: explain that the durable path is
  `aroom workflow run fm_java_17_to_21` from the target repository.
- `apply` or `plan`: direct the user to the durable workflow. Do not reproduce
  the old monolithic edit flow because workflow human gates provide the
  persisted exact-diff approval boundary.

## Operating contract

- Use `glob_files`, `grep`, and `read_file` for static discovery.
- Never use `write_file` or `edit_file` in this interactive facade.
- Use `ask_user` before `bash` in interactive `validate` mode.
- Treat wrappers and build scripts as repository code.
- Do not use `save_memory` or `run_agent`.
- Do not install tools, commit, push, deploy, or update external systems.
- Redact secrets and mark unresolved facts `UNRESOLVED`.

For an implementation request, return the exact command:

```text
aroom workflow run fm_java_17_to_21
```

Mention optional parameters only when useful:

```text
--param change_scope=required
--param validation_profile=standard
```

The workflow owns assessment, proposal persistence, human edit gates,
application, command permission, validation, bounded repair, and reporting.


---
name: java-17-to-21-apply-approved
description: Apply only a persisted Java 17-to-21 proposal or repair patch after a preceding durable workflow human gate has approved that exact patch.
---

# Apply an approved migration patch

Follow the Java modernization policy supplied by this pack.

Invocation arguments: {args}

## Preconditions

- The workflow invocation identifies an approved patch and its dossier.
- A top-level workflow `human_gate` immediately preceded this phase.
- Read the complete dossier and patch with `read_file`.
- Re-read every target file and verify that every original hunk still matches.
- If the patch is absent, incomplete, ambiguous, outside the repository, or
  stale, do not edit anything.

## Contract

- Use `glob_files`, `grep`, and `read_file` for precondition checks.
- Use `edit_file` or `write_file` only for paths explicitly present in the
  approved patch.
- Never broaden, reinterpret, regenerate, or partially repair the patch.
- Do not use `bash`, `ask_user`, `save_memory`, or `run_agent`.
- Do not edit workflow state artifacts except to preserve an application
  result in the named dossier when explicitly requested.
- Do not commit, push, deploy, install tools, or modify external systems.

Apply the exact patch in reversible file-sized batches. Stop on the first
precondition or edit failure and report which files remain untouched.

Begin the final response with exactly one marker:

```text
APPLY_STATUS: APPLIED
APPLY_STATUS: STALE
APPLY_STATUS: FAILED
```

Do not emit assistant commentary before or between tool calls. After tool work
is complete, make the marker the first assistant text in this phase so the
workflow can route deterministically.

Then list changed files, approved patch path, applied suggestion IDs, and
rollback instructions. Do not run validation commands.


---
name: java-17-to-21-assess
description: Perform the read-mostly assessment phase of the durable Java 17-to-21 workflow, including Maven or Gradle discovery, readiness findings, modernization candidates, and complete release-delta dispositions.
---

# Assess Java 17 to 21 readiness

Follow all four Java 17-to-21 instructions supplied by this pack.

Invocation arguments: {args}

## Contract

- Interpret arguments as `repository_path` and `state_prefix`.
- Use `glob_files`, `grep`, and `read_file` for repository discovery.
- Do not use `bash`, `edit_file`, `save_memory`, or `run_agent`.
- Do not modify source, build, CI, container, or runtime configuration.
- `write_file` is permitted only for the workflow assessment artifact named
  `<state_prefix>-assessment.md`.
- Inspect only the supplied repository. Redact secrets.
- Mark unavailable effective values and compatibility evidence `UNRESOLVED`.

## Work

1. Identify Maven, Gradle, nested modules, Java configuration, CI, containers,
   runtime flags, dependencies, plugins, internal APIs, removed/deprecated
   APIs, charset/locale risks, agents, native integration, and modernization
   patterns.
2. Classify findings as `REQUIRED`, `ADVISORY`, `UNRESOLVED`,
   `MODERNIZATION_CANDIDATE`, or `EXPERIMENTAL`, with safety class, evidence,
   confidence, consequences, rollback, and required validation.
3. Give every item in `java-17-to-21-release-delta` a disposition.
4. Write the complete report to `<state_prefix>-assessment.md`.

Begin the final response with exactly one marker:

```text
ASSESSMENT_STATUS: CHANGES_REQUIRED
ASSESSMENT_STATUS: READY_STATIC
ASSESSMENT_STATUS: BLOCKED
ASSESSMENT_STATUS: UNSUPPORTED
```

Do not emit assistant commentary before or between tool calls. After tool work
is complete, make the marker the first assistant text in this phase so the
workflow can route deterministically.

Then state the artifact path, executive summary, required findings, unresolved
risks, and counts by classification. `READY_STATIC` never means runtime
validation passed.


---
name: java-17-to-21-propose
description: Convert a Java 17-to-21 assessment into a selected-scope, exact unified diff and approval dossier for the durable migration workflow without editing repository implementation files.
---

# Propose an exact migration diff

Follow the Java modernization policy and pattern catalog supplied by this pack.

Invocation arguments: {args}

## Contract

- Read `<state_prefix>-assessment.md` and re-read every target file.
- Respect `change_scope`: `required`, `safe`, `modernize`, or `all`.
- Use `glob_files`, `grep`, and `read_file` to verify evidence.
- Do not use `bash`, `edit_file`, `save_memory`, or `run_agent`.
- `write_file` may create only `<state_prefix>-proposal.md` and
  `<state_prefix>-proposal.patch`.
- Do not change source, build, dependency, CI, container, or runtime files.
- Never invent a diff when the original text or effective configuration is
  unresolved.

## Proposal

For every included suggestion record its stable ID, classification, safety
class, evidence, assumptions, affected modules, expected benefit, warnings,
failure consequences, rollback, and validation.

Write one complete unified diff containing only the selected scope to
`<state_prefix>-proposal.patch`. Write the approval dossier, proposal file
paths, and the exact ordered target-file list to
`<state_prefix>-proposal.md`.

Keep Design Sensitive and Experimental changes out unless the supplied scope
explicitly includes them. `all` counts as experimental opt-in only after the
human gate displays that consequence.

Begin the final response with exactly one marker:

```text
PROPOSAL_STATUS: PROPOSED
PROPOSAL_STATUS: NO_CHANGES
PROPOSAL_STATUS: BLOCKED
```

Do not emit assistant commentary before or between tool calls. After tool work
is complete, make the marker the first assistant text in this phase so the
workflow can route deterministically.

For `PROPOSED`, immediately list both artifact paths, included suggestion IDs,
safety classes, target files, and the strongest failure consequence. Tell the
operator that approval applies only to the complete persisted patch.


---
name: java-17-to-21-repair
description: Diagnose a failed Java 17-to-21 validation round and prepare one minimal exact repair patch for a separate durable human approval gate.
---

# Propose one bounded repair

Follow the Java modernization policy supplied by this pack.

Invocation arguments: {args}

## Contract

- Read the failed validation artifact, prior approved patch, and current target
  files.
- Distinguish migration, implementation, pre-existing, environmental,
  permission, and unresolved failures.
- Propose only a minimal repository-local repair for migration-related or
  implementation-related failure.
- Use `glob_files`, `grep`, and `read_file` for diagnosis.
- Do not use `bash`, `edit_file`, `ask_user`, `save_memory`, or `run_agent`.
- `write_file` may create only `<state_prefix>-repair-<round>.md` and
  `<state_prefix>-repair-<round>.patch`.
- Never silently apply a repair or broaden the original change scope.

The repair dossier must include evidence, safety class, assumptions, exact
unified diff, failure consequences, rollback, and validation commands. For
pre-existing, environmental, permission, or unresolved failure, write a
dossier explaining why no responsible source repair exists and do not create a
non-empty patch.

Begin the final response with exactly one marker:

```text
REPAIR_STATUS: PROPOSED
REPAIR_STATUS: NOT_APPLICABLE
REPAIR_STATUS: BLOCKED
```

Do not emit assistant commentary before or between tool calls. After tool work
is complete, make the marker the first assistant text in this phase so the
workflow can route deterministically.

For `PROPOSED`, list the exact repair dossier and patch paths.


---
name: java-17-to-21-report
description: Produce the final evidence-based report for a durable Java 17-to-21 workflow run from persisted assessment, proposal, application, validation, and repair artifacts.
---

# Report the durable migration result

Follow all Java 17-to-21 instructions supplied by this pack.

Invocation arguments: {args}

Use `glob_files` and `read_file` to collect all matching workflow state
artifacts and inspect relevant current repository files. Do not use `bash`,
`grep`, `write_file`, `edit_file`, `ask_user`, `save_memory`, or `run_agent`.

Separate:

- readiness and modernization findings
- selected, approved, rejected, applied, and remaining changes
- accepted warnings and failure consequences by safety class
- exact changed files and rollback
- commands, permission outcomes, and validation results
- each repair round and its approval outcome
- release-delta coverage and checks not performed

Report one overall status: `READY`, `CHANGES REQUIRED`, `BLOCKED`,
`PARTIALLY VALIDATED`, or `UNSUPPORTED`. Never report `READY` unless required
runtime validation passed and no required or high-risk unresolved findings
remain. Optional modernization may remain when readiness is `READY`.

Begin the response with:

```text
WORKFLOW_RESULT: <overall status>
```


---
name: java-17-to-21-validate
description: Validate a Java 17-to-21 migration with repository-appropriate Java, Maven or Gradle, test, bytecode, and delivery checks under Anteroom workflow tool approval.
---

# Validate the migration

Follow the validation section of the Java baseline supplied by this pack.

Invocation arguments: {args}

## Contract

- Interpret arguments as repository path, state prefix, validation round, and
  validation profile.
- Re-read build files and project documentation before choosing commands.
- Show each exact command, working directory, purpose, possible downloads and
  side effects before invoking `bash`.
- Let Anteroom's workflow approval pause the run. Never treat workflow start or
  edit approval as command permission.
- Respect the recorded decision: `Allow Once` authorizes one displayed request,
  `Allow for Session` remains limited to this repository and migration, and
  rejection leaves validation `BLOCKED`.
- Use only repository-documented wrappers when present.
- Do not install tools, edit implementation files, commit, push, or deploy.
- Use `write_file` only for
  `<state_prefix>-validation-<round>.md`.
- Do not use `edit_file`, `ask_user`, `save_memory`, or `run_agent`.

For `validation_profile=none`, perform static post-edit checks only and report
`BLOCKED` because runtime validation was intentionally skipped. For `standard`,
check the Java/Javac environment, wrapper/tool version, compile/package, normal
tests, and practical `jdeps`/`jdeprscan` and delivery alignment. Do not run an
application or contact external services unless repository documentation and
the displayed command make that scope explicit.

Classify every failure as migration-related, implementation-related,
pre-existing, environmental, permission-related, or unresolved. Persist the
complete command log and conclusion.

Begin the final response with exactly one marker:

```text
VALIDATION_STATUS: PASS
VALIDATION_STATUS: FAIL
VALIDATION_STATUS: BLOCKED
```

Do not emit assistant commentary before or between tool calls. After tool work
is complete, make the marker the first assistant text in this phase so workflow
routing remains reliable.



kind: workflow
id: fm_java_17_to_21
version: 0.4.0
description: Durable Java 17-to-21 migration with exact-diff gates and two bounded repair rounds.
tags:
  - java
  - migration
  - human-gated

inputs: {}

params:
  repository_path:
    default: "."
  change_scope:
    default: "required"
  validation_profile:
    default: "standard"
  state_prefix:
    default: ".anteroom-java-21"

policies:
  inject_rules: true
  inject_conventions: true
  budget:
    max_steps: 18
    max_tokens: 300000
    max_duration_seconds: 10800

steps:
  - id: assess
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-assess
    skill_args: "repository_path={repository_path} state_prefix={state_prefix}"
    working_dir: "{repository_path}"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
    max_iterations: 30
    timeout: 1200

  - id: propose
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-propose
    skill_args: "repository_path={repository_path} state_prefix={state_prefix} change_scope={change_scope}"
    working_dir: "{repository_path}"
    context_from:
      - step: assess
        field: result_summary
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
    max_iterations: 30
    timeout: 1200

  - id: approve_initial_diff
    type: human_gate
    prompt: >-
      Review the complete persisted proposal dossier and patch named by the
      proposal step. Approve only that exact patch. Selection does not
      authorize any other edit, and edit approval does not authorize build or
      test commands.
    context_from:
      - step: propose
        field: result_summary
    when:
      step: propose
      field: result_summary
      contains: "PROPOSAL_STATUS: PROPOSED"
    options:
      - id: approve_diff
        label: Approve exact diff
        outcome: continue
      - id: stop
        label: Stop without source edits
        outcome: stop
        stop_reason: initial_diff_not_approved

  - id: apply_initial_diff
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-apply-approved
    skill_args: >-
      repository_path={repository_path}
      dossier={state_prefix}-proposal.md
      approved_patch={state_prefix}-proposal.patch
    working_dir: "{repository_path}"
    context_from:
      - step: propose
        field: result_summary
    when:
      step: propose
      field: result_summary
      contains: "PROPOSAL_STATUS: PROPOSED"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
      - edit_file
    max_iterations: 30
    timeout: 1200

  - id: validate_1
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-validate
    skill_args: >-
      repository_path={repository_path} state_prefix={state_prefix}
      validation_round=1 validation_profile={validation_profile}
    working_dir: "{repository_path}"
    context_from:
      - step: assess
        field: result_summary
      - step: apply_initial_diff
        field: result_summary
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
      - bash
    max_iterations: 35
    timeout: 2400
    continue_on:
      - failed

  - id: propose_repair_1
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-repair
    skill_args: >-
      repository_path={repository_path} state_prefix={state_prefix}
      failed_validation_round=1 repair_round=1
    working_dir: "{repository_path}"
    context_from:
      - step: validate_1
        field: result_summary
    when:
      step: validate_1
      field: result_summary
      contains: "VALIDATION_STATUS: FAIL"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
    max_iterations: 25
    timeout: 900

  - id: approve_repair_1
    type: human_gate
    prompt: >-
      Review the complete repair-1 dossier and patch named by the repair step.
      Approve only that exact repair patch. This approval does not authorize
      validation commands.
    context_from:
      - step: propose_repair_1
        field: result_summary
    when:
      step: propose_repair_1
      field: result_summary
      contains: "REPAIR_STATUS: PROPOSED"
    options:
      - id: approve_repair
        label: Approve repair diff
        outcome: continue
      - id: stop
        label: Stop and report failure
        outcome: stop
        stop_reason: repair_1_not_approved

  - id: apply_repair_1
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-apply-approved
    skill_args: >-
      repository_path={repository_path}
      dossier={state_prefix}-repair-1.md
      approved_patch={state_prefix}-repair-1.patch
    working_dir: "{repository_path}"
    context_from:
      - step: propose_repair_1
        field: result_summary
    when:
      step: propose_repair_1
      field: result_summary
      contains: "REPAIR_STATUS: PROPOSED"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
      - edit_file
    max_iterations: 25
    timeout: 900

  - id: validate_2
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-validate
    skill_args: >-
      repository_path={repository_path} state_prefix={state_prefix}
      validation_round=2 validation_profile={validation_profile}
    working_dir: "{repository_path}"
    context_from:
      - step: apply_repair_1
        field: result_summary
    when:
      step: apply_repair_1
      field: result_summary
      contains: "APPLY_STATUS: APPLIED"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
      - bash
    max_iterations: 35
    timeout: 2400
    continue_on:
      - failed

  - id: propose_repair_2
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-repair
    skill_args: >-
      repository_path={repository_path} state_prefix={state_prefix}
      failed_validation_round=2 repair_round=2
    working_dir: "{repository_path}"
    context_from:
      - step: validate_2
        field: result_summary
    when:
      step: validate_2
      field: result_summary
      contains: "VALIDATION_STATUS: FAIL"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
    max_iterations: 25
    timeout: 900

  - id: approve_repair_2
    type: human_gate
    prompt: >-
      Review the complete repair-2 dossier and patch named by the repair step.
      This is the final automatic repair round. Approve only that exact patch.
    context_from:
      - step: propose_repair_2
        field: result_summary
    when:
      step: propose_repair_2
      field: result_summary
      contains: "REPAIR_STATUS: PROPOSED"
    options:
      - id: approve_repair
        label: Approve final repair
        outcome: continue
      - id: stop
        label: Stop and report failure
        outcome: stop
        stop_reason: repair_2_not_approved

  - id: apply_repair_2
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-apply-approved
    skill_args: >-
      repository_path={repository_path}
      dossier={state_prefix}-repair-2.md
      approved_patch={state_prefix}-repair-2.patch
    working_dir: "{repository_path}"
    context_from:
      - step: propose_repair_2
        field: result_summary
    when:
      step: propose_repair_2
      field: result_summary
      contains: "REPAIR_STATUS: PROPOSED"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
      - edit_file
    max_iterations: 25
    timeout: 900

  - id: validate_3
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-validate
    skill_args: >-
      repository_path={repository_path} state_prefix={state_prefix}
      validation_round=3 validation_profile={validation_profile}
    working_dir: "{repository_path}"
    context_from:
      - step: apply_repair_2
        field: result_summary
    when:
      step: apply_repair_2
      field: result_summary
      contains: "APPLY_STATUS: APPLIED"
    tools:
      - glob_files
      - grep
      - read_file
      - write_file
      - bash
    max_iterations: 35
    timeout: 2400
    continue_on:
      - failed

  - id: final_report
    type: runner
    runner: cli_claude
    skill_name: java-17-to-21-report
    skill_args: "repository_path={repository_path} state_prefix={state_prefix}"
    working_dir: "{repository_path}"
    context_from:
      - step: assess
        field: result_summary
      - step: propose
        field: result_summary
      - step: apply_initial_diff
        field: result_summary
      - step: validate_1
        field: result_summary
      - step: propose_repair_1
        field: result_summary
      - step: apply_repair_1
        field: result_summary
      - step: validate_2
        field: result_summary
      - step: propose_repair_2
        field: result_summary
      - step: apply_repair_2
        field: result_summary
      - step: validate_3
        field: result_summary
    tools:
      - glob_files
      - read_file
    max_iterations: 25
    timeout: 900
