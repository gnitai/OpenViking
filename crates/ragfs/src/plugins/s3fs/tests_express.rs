//! Unit tests for S3 Express One Zone support.
//!
//! Covers: bucket-name parsing, config validation, and `S3Client` construction
//! when `express: true` is set. The live integration test
//! (`test_express_roundtrip_live`) is `#[ignore]`d by default and only runs
//! against real AWS when the `s3-express-live` feature is enabled and the
//! required env vars are set.

use std::collections::HashMap;

use super::client::{parse_express_bucket_name, validate_express_config, S3Client};
use super::{S3FSPlugin, ServicePlugin};
use crate::core::{ConfigValue, PluginConfig};

// ─── helpers ────────────────────────────────────────────────────────────────

fn cfg(pairs: &[(&str, ConfigValue)]) -> PluginConfig {
    let mut params = HashMap::new();
    for (k, v) in pairs {
        params.insert((*k).to_string(), v.clone());
    }
    PluginConfig {
        name: "s3fs".to_string(),
        mount_path: "/s3".to_string(),
        params,
    }
}

fn s(v: &str) -> ConfigValue {
    ConfigValue::String(v.to_string())
}

fn b(v: bool) -> ConfigValue {
    ConfigValue::Bool(v)
}

// ─── 1. parse_express_bucket_name: positive cases ───────────────────────────

#[test]
fn test_parse_express_bucket_name_valid() {
    assert_eq!(
        parse_express_bucket_name("my-cache--use1-az4--x-s3"),
        Some("use1-az4")
    );
    assert_eq!(
        parse_express_bucket_name("alpha--usw2-az1--x-s3"),
        Some("usw2-az1")
    );
    assert_eq!(
        parse_express_bucket_name("beta--apne1-az3--x-s3"),
        Some("apne1-az3")
    );
    // base name with internal hyphens is fine
    assert_eq!(
        parse_express_bucket_name("ov-test-cache--use1-az4--x-s3"),
        Some("use1-az4")
    );
}

// ─── 2. parse_express_bucket_name: negative cases ───────────────────────────

#[test]
fn test_parse_express_bucket_name_rejects_non_express() {
    assert_eq!(parse_express_bucket_name("regular-bucket"), None);
    assert_eq!(parse_express_bucket_name(""), None);
    // missing AZ-ID segment
    assert_eq!(parse_express_bucket_name("only--x-s3"), None);
    // wrong suffix
    assert_eq!(parse_express_bucket_name("cache--use1-az4--xs3"), None);
    // malformed AZ id
    assert_eq!(parse_express_bucket_name("cache--notaz--x-s3"), None);
}

// ─── 3. validate_express_config: AZ ID must match suffix in bucket name ─────

#[test]
fn test_express_bucket_name_az_must_match_config() {
    assert!(validate_express_config("cache--use1-az4--x-s3", "use1-az4").is_ok());
    assert!(validate_express_config("cache--use1-az4--x-s3", "usw2-az1").is_err());
}

// ─── 4. validate_express_config / plugin: requires AZ id ────────────────────

#[tokio::test]
async fn test_validate_express_requires_az_id() {
    let plugin = S3FSPlugin::new();
    let config = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        // availability_zone_id intentionally absent
    ]);
    let err = plugin
        .validate(&config)
        .await
        .expect_err("missing availability_zone_id must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("availability_zone_id"),
        "error should mention availability_zone_id, got: {msg}"
    );
}

// ─── 5. AZ-ID format guard (rejects zone *names* like "us-east-1a") ─────────

#[tokio::test]
async fn test_validate_express_az_id_format() {
    let plugin = S3FSPlugin::new();

    // Zone *name* (us-east-1a) is not an AZ *ID* — reject.
    let bad = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        ("availability_zone_id", s("us-east-1a")),
    ]);
    assert!(
        plugin.validate(&bad).await.is_err(),
        "zone name (us-east-1a) must be rejected — AZ ID required"
    );

    // Proper AZ ID
    let good = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        ("availability_zone_id", s("use1-az4")),
    ]);
    assert!(plugin.validate(&good).await.is_ok());
}

// ─── 6. Express forbids force-path-style (zonal endpoint is vhost only) ─────

#[tokio::test]
async fn test_validate_express_rejects_path_style() {
    let plugin = S3FSPlugin::new();
    let config = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        ("availability_zone_id", s("use1-az4")),
        ("use_path_style", b(true)),
    ]);
    let err = plugin
        .validate(&config)
        .await
        .expect_err("use_path_style=true must be rejected under express");
    let msg = format!("{err}");
    assert!(
        msg.contains("path") || msg.contains("virtual"),
        "error should explain addressing constraint, got: {msg}"
    );
}

// ─── 7. Express forbids custom endpoint ─────────────────────────────────────

#[tokio::test]
async fn test_validate_express_rejects_custom_endpoint() {
    let plugin = S3FSPlugin::new();
    let config = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        ("availability_zone_id", s("use1-az4")),
        ("endpoint", s("https://example.invalid")),
    ]);
    assert!(
        plugin.validate(&config).await.is_err(),
        "custom endpoint must be rejected under express"
    );
}

// ─── 8. Express forbids disable_batch_delete (only relevant to OSS-like) ────

#[tokio::test]
async fn test_validate_express_rejects_disable_batch_delete() {
    let plugin = S3FSPlugin::new();
    let config = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        ("availability_zone_id", s("use1-az4")),
        ("disable_batch_delete", b(true)),
    ]);
    assert!(plugin.validate(&config).await.is_err());
}

// ─── 9. Express rejects nonempty directory_marker_mode; empty/none ok ──────

#[tokio::test]
async fn test_validate_express_rejects_nonempty_marker() {
    let plugin = S3FSPlugin::new();

    let bad = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        ("express", b(true)),
        ("availability_zone_id", s("use1-az4")),
        ("directory_marker_mode", s("nonempty")),
    ]);
    assert!(plugin.validate(&bad).await.is_err());

    for mode in &["empty", "none"] {
        let good = cfg(&[
            ("bucket", s("cache--use1-az4--x-s3")),
            ("express", b(true)),
            ("availability_zone_id", s("use1-az4")),
            ("directory_marker_mode", s(mode)),
        ]);
        assert!(
            plugin.validate(&good).await.is_ok(),
            "marker_mode={mode} should be accepted under express"
        );
    }
}

// ─── 10. Bucket has --x-s3 suffix but caller didn't opt in: guardrail err ───

#[tokio::test]
async fn test_validate_express_bucket_suffix_without_flag() {
    let plugin = S3FSPlugin::new();
    let config = cfg(&[
        ("bucket", s("cache--use1-az4--x-s3")),
        // express flag absent (treated as false)
    ]);
    let err = plugin
        .validate(&config)
        .await
        .expect_err("express-shaped bucket name without express=true must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("express"),
        "error should prompt user to set express: true, got: {msg}"
    );
}

// ─── 11. Existing non-Express configs still validate (regression guard) ─────

#[tokio::test]
async fn test_validate_non_express_unchanged() {
    let plugin = S3FSPlugin::new();

    // Plain AWS S3 config
    let aws = cfg(&[("bucket", s("regular-bucket")), ("region", s("us-east-1"))]);
    assert!(plugin.validate(&aws).await.is_ok());

    // MinIO-style config — endpoint + path-style remain valid when express is false
    let minio = cfg(&[
        ("bucket", s("test")),
        ("endpoint", s("http://localhost:9000")),
        ("use_path_style", b(true)),
        ("access_key_id", s("minioadmin")),
        ("secret_access_key", s("minioadmin")),
    ]);
    assert!(plugin.validate(&minio).await.is_ok());

    // TOS-style config — nonempty marker still valid when express is false
    let tos = cfg(&[
        ("bucket", s("my-tos-bucket")),
        ("endpoint", s("https://tos-cn-beijing.volces.com")),
        ("directory_marker_mode", s("nonempty")),
        ("use_path_style", b(false)),
    ]);
    assert!(plugin.validate(&tos).await.is_ok());
}

// ─── 12. S3Client::new with express config sets flags ───────────────────────

#[tokio::test]
async fn test_s3client_new_express_sets_flags() {
    let mut params = HashMap::new();
    params.insert(
        "bucket".to_string(),
        s("cache--use1-az4--x-s3"),
    );
    params.insert("region".to_string(), s("us-east-1"));
    params.insert("express".to_string(), b(true));
    params.insert("availability_zone_id".to_string(), s("use1-az4"));
    // Provide explicit creds so the SDK doesn't probe the environment.
    params.insert("access_key_id".to_string(), s("AKIAIOSFODNN7EXAMPLE"));
    params.insert(
        "secret_access_key".to_string(),
        s("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    );

    let client = S3Client::new(&params)
        .await
        .expect("S3Client::new should succeed for express config");

    assert!(client.is_express(), "is_express() should be true");
    assert_eq!(client.availability_zone_id(), "use1-az4");
    assert_eq!(client.bucket(), "cache--use1-az4--x-s3");
}

// ─── 13. S3Client::new without express keeps existing behavior ──────────────

#[tokio::test]
async fn test_s3client_new_default_non_express() {
    let mut params = HashMap::new();
    params.insert("bucket".to_string(), s("regular-bucket"));
    params.insert("region".to_string(), s("us-east-1"));
    params.insert("access_key_id".to_string(), s("AKIAIOSFODNN7EXAMPLE"));
    params.insert(
        "secret_access_key".to_string(),
        s("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
    );

    let client = S3Client::new(&params)
        .await
        .expect("S3Client::new should succeed for non-express config");

    assert!(!client.is_express());
    assert_eq!(client.availability_zone_id(), "");
}

// ─── 14. readme() documents the Express gaps so callers see them ────────────

#[test]
fn test_express_readme_contains_unsupported_list() {
    let plugin = S3FSPlugin::new();
    let readme = plugin.readme().to_lowercase();
    for needle in &[
        "express",
        "directory bucket",
        "no versioning",
        "no object tags",
        "etag is opaque",
        "availability_zone_id",
    ] {
        assert!(
            readme.contains(needle),
            "readme() should mention '{needle}', current readme:\n{}",
            plugin.readme()
        );
    }
}

// ─── 15. Live integration roundtrip (gated, ignored by default) ─────────────

/// End-to-end roundtrip against a real S3 Express directory bucket.
///
/// Skipped unless all of these env vars are set:
///   - `AWS_TEST_S3_EXPRESS_BUCKET`  (e.g. `ov-test-express--use1-az4--x-s3`)
///   - `AWS_TEST_S3_EXPRESS_AZ`      (e.g. `use1-az4`)
///   - `AWS_TEST_S3_EXPRESS_REGION`  (e.g. `us-east-1`)
///
/// Run with:
///   `cargo test -p ragfs --features s3,s3-express-live -- --ignored test_express_roundtrip_live`
#[tokio::test]
#[cfg_attr(not(feature = "s3-express-live"), ignore)]
async fn test_express_roundtrip_live() {
    let bucket = match std::env::var("AWS_TEST_S3_EXPRESS_BUCKET") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("AWS_TEST_S3_EXPRESS_BUCKET not set; skipping live express test");
            return;
        }
    };
    let az = match std::env::var("AWS_TEST_S3_EXPRESS_AZ") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("AWS_TEST_S3_EXPRESS_AZ not set; skipping live express test");
            return;
        }
    };
    let region = match std::env::var("AWS_TEST_S3_EXPRESS_REGION") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("AWS_TEST_S3_EXPRESS_REGION not set; skipping live express test");
            return;
        }
    };

    let mut params = HashMap::new();
    params.insert("bucket".to_string(), s(&bucket));
    params.insert("region".to_string(), s(&region));
    params.insert("express".to_string(), b(true));
    params.insert("availability_zone_id".to_string(), s(&az));
    // Pass creds through the plugin config so the SDK doesn't fall back to
    // a profile/IMDS resolution that might differ from aws-cli's chain.
    if let Ok(ak) = std::env::var("AWS_ACCESS_KEY_ID") {
        params.insert("access_key_id".to_string(), s(&ak));
    }
    if let Ok(sk) = std::env::var("AWS_SECRET_ACCESS_KEY") {
        params.insert("secret_access_key".to_string(), s(&sk));
    }

    let client = S3Client::new(&params)
        .await
        .expect("S3Client::new should succeed for live express config");

    let key = format!("ragfs-express-roundtrip/{}.txt", uuid::Uuid::new_v4());
    let body = b"hello s3 express".to_vec();

    client
        .put_object(&key, body.clone())
        .await
        .expect("put_object failed");

    let head = client
        .head_object(&key)
        .await
        .expect("head_object failed")
        .expect("head_object returned None for an object we just PUT");
    assert_eq!(head.size as usize, body.len());
    assert!(!head.key.is_empty(), "ETag/key must be populated");

    let read = client.get_object(&key).await.expect("get_object failed");
    assert_eq!(read, body);

    client
        .delete_object(&key)
        .await
        .expect("delete_object failed");
}

/// Validates the atomic `RenameObject` path (and the `rename_source` header
/// format) against a real S3 Express directory bucket: PUT a source object,
/// rename it to a different prefix, assert the source is gone and the
/// destination has the original bytes.
///
/// Same env-var gating as `test_express_roundtrip_live`.
#[tokio::test]
#[cfg_attr(not(feature = "s3-express-live"), ignore)]
async fn test_express_rename_object_live() {
    let bucket = match std::env::var("AWS_TEST_S3_EXPRESS_BUCKET") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("AWS_TEST_S3_EXPRESS_BUCKET not set; skipping live express test");
            return;
        }
    };
    let az = match std::env::var("AWS_TEST_S3_EXPRESS_AZ") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("AWS_TEST_S3_EXPRESS_AZ not set; skipping live express test");
            return;
        }
    };
    let region = match std::env::var("AWS_TEST_S3_EXPRESS_REGION") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("AWS_TEST_S3_EXPRESS_REGION not set; skipping live express test");
            return;
        }
    };

    let mut params = HashMap::new();
    params.insert("bucket".to_string(), s(&bucket));
    params.insert("region".to_string(), s(&region));
    params.insert("express".to_string(), b(true));
    params.insert("availability_zone_id".to_string(), s(&az));
    if let Ok(ak) = std::env::var("AWS_ACCESS_KEY_ID") {
        params.insert("access_key_id".to_string(), s(&ak));
    }
    if let Ok(sk) = std::env::var("AWS_SECRET_ACCESS_KEY") {
        params.insert("secret_access_key".to_string(), s(&sk));
    }

    let client = S3Client::new(&params)
        .await
        .expect("S3Client::new should succeed for live express config");

    let id = uuid::Uuid::new_v4();
    let src = format!("ragfs-express-rename/{}/src.txt", id);
    let dst = format!("ragfs-express-rename/{}/sub/dst.txt", id);
    let body = b"rename me atomically".to_vec();

    client.put_object(&src, body.clone()).await.expect("put_object failed");

    client
        .rename_object(&src, &dst)
        .await
        .expect("rename_object failed (check rename_source header format)");

    let src_head = client.head_object(&src).await.expect("head src failed");
    assert!(src_head.is_none(), "source object should be gone after rename");

    let read = client.get_object(&dst).await.expect("get dst failed");
    assert_eq!(read, body, "destination must hold original bytes");

    client.delete_object(&dst).await.ok();
}
